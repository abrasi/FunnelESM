"""Minimal ESM-2 variants with inter-layer sequence reductions."""

from __future__ import annotations

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
    EsmModel,
    eager_attention_forward,
)

from funnel_esm.pooling import MaskedMeanPool1d


class FunnelEsmModel(EsmModel):
    """Reuse Hugging Face ESM layers and reduce sequence length."""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        pool_after_layers = getattr(
            config,
            "funnel_pool_after_layers",
            None,
        )
        if pool_after_layers is None:
            pool_after_layers = [
                getattr(
                    config,
                    "funnel_pool_after_layer",
                    config.num_hidden_layers // 2,
                )
            ]

        self.pool_after_layers = tuple(pool_after_layers)

        if not self.pool_after_layers:
            raise ValueError("funnel_pool_after_layers cannot be empty")
        if len(set(self.pool_after_layers)) != len(
            self.pool_after_layers
        ):
            raise ValueError("funnel_pool_after_layers contains duplicates")
        if tuple(sorted(self.pool_after_layers)) != self.pool_after_layers:
            raise ValueError("funnel_pool_after_layers must be sorted")
        if any(
            layer < 1 or layer >= config.num_hidden_layers
            for layer in self.pool_after_layers
        ):
            raise ValueError(
                "Each pooling layer must be between 1 and "
                "num_hidden_layers - 1"
            )

        self.sequence_pooler = MaskedMeanPool1d()

    def _prepare_layer_inputs(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        extended_attention_mask, extended_encoder_attention_mask = (
            self._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                embedding_output=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=None,
            )
        )

        if self.position_embedding_type == "rotary":
            position_embeddings = self.rotary_embeddings(
                hidden_states,
                position_ids,
            )
        else:
            position_embeddings = None

        return (
            extended_attention_mask,
            extended_encoder_attention_mask,
            position_embeddings,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            hidden_states = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            hidden_states = inputs_embeds

        batch_size, sequence_length = hidden_states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.long,
                device=hidden_states.device,
            )

        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0)

        (
            extended_attention_mask,
            extended_encoder_attention_mask,
            position_embeddings,
        ) = self._prepare_layer_inputs(
            hidden_states,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        for layer_number, layer_module in enumerate(
            self.encoder.layer,
            start=1,
        ):
            hidden_states = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_encoder_attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if layer_number in self.pool_after_layers:
                hidden_states, attention_mask, position_ids = (
                    self.sequence_pooler(
                        hidden_states,
                        attention_mask,
                        position_ids,
                    )
                )
                (
                    extended_attention_mask,
                    extended_encoder_attention_mask,
                    position_embeddings,
                ) = self._prepare_layer_inputs(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )

        hidden_states = self.encoder.emb_layer_norm_after(hidden_states)
        pooled_output = (
            self.pooler(hidden_states)
            if self.pooler is not None
            else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )


class FunnelEsmForSequenceClassification(EsmForSequenceClassification):
    """Sequence classifier using :class:`FunnelEsmModel` as its encoder."""

    def __init__(self, config):
        super().__init__(config)
        self.esm = FunnelEsmModel(config, add_pooling_layer=False)


def _apply_rope(
    states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply Hugging Face ESM RoPE to one tensor."""
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    original_dtype = states.dtype
    states = states.float()
    cos = cos.float()
    sin = sin.float()
    first_half, second_half = states.chunk(2, dim=-1)
    rotated = torch.cat((-second_half, first_half), dim=-1)
    return (states * cos + rotated * sin).to(original_dtype)


class FunnelEsmPoolQueryModel(FunnelEsmModel):
    """ESM-2 with FunnelTransformer-style pool-query-only transitions."""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        pool_stride = getattr(config, "funnel_pool_stride", 2)
        if pool_stride != 2:
            raise ValueError("pool-query-only currently requires stride 2")

    def _pool_query_layer(
        self,
        layer_module,
        query_states,
        key_value_states,
        query_mask,
        key_value_mask,
        query_positions,
        key_value_positions,
        **kwargs,
    ):
        """Run one ESM layer with pooled Q and unpooled K/V."""
        attention = layer_module.attention
        self_attention = attention.self

        query_input = attention.LayerNorm(query_states)
        key_value_input = attention.LayerNorm(key_value_states)

        batch_size, query_length, _ = query_input.shape
        key_value_length = key_value_input.shape[1]
        num_heads = self_attention.num_attention_heads
        head_size = self_attention.attention_head_size

        query = self_attention.query(query_input).view(
            batch_size,
            query_length,
            num_heads,
            head_size,
        )
        key = self_attention.key(key_value_input).view(
            batch_size,
            key_value_length,
            num_heads,
            head_size,
        )
        value = self_attention.value(key_value_input).view(
            batch_size,
            key_value_length,
            num_heads,
            head_size,
        )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # ESM scales Q before applying RoPE.
        query = query * (head_size**-0.5)

        if self.position_embedding_type == "rotary":
            query_rope = self.rotary_embeddings(
                query_states,
                query_positions,
            )
            key_rope = self.rotary_embeddings(
                key_value_states,
                key_value_positions,
            )
            query = _apply_rope(query, query_rope)
            key = _apply_rope(key, key_rope)

        _, attention_mask = self._create_attention_masks(
            attention_mask=query_mask,
            encoder_attention_mask=key_value_mask,
            embedding_output=query_states,
            encoder_hidden_states=key_value_states,
            past_key_values=None,
        )
        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
        )
        attention_output, _ = attention_interface(
            self_attention,
            query,
            key,
            value,
            attention_mask,
            dropout=(
                0.0
                if not self.training
                else self_attention.dropout
            ),
            scaling=self_attention.scaling,
            **kwargs,
        )
        attention_output = attention_output.reshape(
            batch_size,
            query_length,
            num_heads * head_size,
        )

        attention_output = attention.output(
            attention_output,
            query_states,
        )
        layer_output = layer_module.feed_forward_chunk(attention_output)

        # Padding queries are not used as K/V in subsequent layers.
        return layer_output * query_mask.unsqueeze(-1).to(
            layer_output.dtype
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "Specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            hidden_states = self.embeddings(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        else:
            hidden_states = inputs_embeds

        batch_size, sequence_length = hidden_states.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, sequence_length),
                dtype=torch.long,
                device=hidden_states.device,
            )

        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=hidden_states.device,
            ).unsqueeze(0)

        (
            extended_attention_mask,
            extended_encoder_attention_mask,
            position_embeddings,
        ) = self._prepare_layer_inputs(
            hidden_states,
            attention_mask,
            position_ids,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        transition_layers = {
            layer_number + 1
            for layer_number in self.pool_after_layers
        }

        for layer_number, layer_module in enumerate(
            self.encoder.layer,
            start=1,
        ):
            if layer_number in transition_layers:
                full_states = hidden_states
                full_mask = attention_mask
                full_positions = position_ids

                (
                    pooled_states,
                    pooled_mask,
                    pooled_positions,
                ) = self.sequence_pooler(
                    full_states,
                    full_mask,
                    full_positions,
                )

                hidden_states = self._pool_query_layer(
                    layer_module=layer_module,
                    query_states=pooled_states,
                    key_value_states=full_states,
                    query_mask=pooled_mask,
                    key_value_mask=full_mask,
                    query_positions=pooled_positions,
                    key_value_positions=full_positions,
                    **kwargs,
                )
                attention_mask = pooled_mask
                position_ids = pooled_positions

                (
                    extended_attention_mask,
                    extended_encoder_attention_mask,
                    position_embeddings,
                ) = self._prepare_layer_inputs(
                    hidden_states,
                    attention_mask,
                    position_ids,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
                continue

            hidden_states = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=extended_encoder_attention_mask,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.encoder.emb_layer_norm_after(hidden_states)
        pooled_output = (
            self.pooler(hidden_states)
            if self.pooler is not None
            else None
        )

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )


class FunnelEsmPoolQueryForSequenceClassification(
    EsmForSequenceClassification
):
    """Sequence classifier with pool-query-only transitions."""

    def __init__(self, config):
        super().__init__(config)
        self.esm = FunnelEsmPoolQueryModel(
            config,
            add_pooling_layer=False,
        )