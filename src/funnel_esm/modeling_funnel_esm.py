"""A minimal ESM-2 variant with one inter-layer sequence reduction."""

from __future__ import annotations

import torch
from transformers.modeling_outputs import (
    BaseModelOutputWithPoolingAndCrossAttentions,
)
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
    EsmModel,
)

from funnel_esm.pooling import MaskedMeanPool1d


class FunnelEsmModel(EsmModel):
    """Reuse Hugging Face ESM layers and reduce sequence length once."""

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        self.pool_after_layer = getattr(
            config,
            "funnel_pool_after_layer",
            config.num_hidden_layers // 2,
        )

        if not 1 <= self.pool_after_layer < config.num_hidden_layers:
            raise ValueError(
                "funnel_pool_after_layer must be between 1 and "
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

            if layer_number == self.pool_after_layer:
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
