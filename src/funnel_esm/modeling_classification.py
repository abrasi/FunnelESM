"""Shared masked-attention classifier for baseline and Funnel ESM."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers import AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
)

from funnel_esm.pooling import MaskedAttentionClassificationHead

ROPE_KEY_MAPPING = {
    "esm.encoder.layer.*.attention.self.rotary_embeddings.inv_freq": "esm.rotary_embeddings.inv_freq",
}


def load_pretrained_classifier(
    model_class, model_name, labels, focal_gamma=1.0, pool_after_layers=(), pooling_type=None
):
    """Load an ESM-2 backbone and initialize only the task classification head."""
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type != "esm" or config.position_embedding_type != "rotary":
        raise ValueError("Choose a pretrained ESM-2 model with rotary positions")
    layers = list(pool_after_layers)
    if layers != sorted(set(layers)) or any(not 1 <= i < config.num_hidden_layers for i in layers):
        raise ValueError(
            f"pool_after_layers must be sorted, unique and between 1 and {config.num_hidden_layers - 1}"
        )
    config.num_labels = len(labels)
    config.id2label = dict(enumerate(labels))
    config.label2id = {label: index for index, label in enumerate(labels)}
    config.problem_type = "multi_label_classification"
    config.deeploc_focal_gamma = focal_gamma
    config.funnel_backbone_name = model_name
    config.funnel_backbone_frozen = True
    config.funnel_pool_after_layers = layers
    if layers:
        config.funnel_pooling_type = pooling_type
        config.funnel_pool_stride = 2
    model, loading = model_class.from_pretrained(
        model_name,
        config=config,
        key_mapping=ROPE_KEY_MAPPING,
        output_loading_info=True,
    )
    missing = [key for key in loading["missing_keys"] if not key.startswith("classifier.")]
    unused = [key for key in loading["unexpected_keys"] if key.startswith("esm.")]
    if missing or unused or loading.get("mismatched_keys"):
        raise ValueError(f"Pretrained backbone did not load completely: {loading}")
    model.base_model.requires_grad_(False)
    model.accepts_loss_kwargs = False
    model.set_label_weights(torch.ones(config.num_labels))
    return model


class MaskedAttentionEsmForSequenceClassification(EsmForSequenceClassification):
    """ESM classifier with learned pooling and weighted focal loss."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = MaskedAttentionClassificationHead(config)
        self.classifier.apply(self._init_weights)
        self.focal_gamma = getattr(
            config,
            "deeploc_focal_gamma",
            1.0,
        )
        if self.focal_gamma < 0:
            raise ValueError("deeploc_focal_gamma cannot be negative")
        label_weights = getattr(
            config,
            "deeploc_label_weights",
            [1.0] * config.num_labels,
        )
        self.register_buffer(
            "label_weights",
            torch.tensor(label_weights, dtype=torch.float32),
            persistent=False,
        )

    def set_label_weights(self, label_weights: torch.Tensor) -> None:
        """Set fold-specific inverse-frequency weights for the loss."""
        if label_weights.shape != (self.config.num_labels,):
            raise ValueError("label_weights must contain one value per label")
        self.label_weights.copy_(
            label_weights.to(
                device=self.label_weights.device,
                dtype=self.label_weights.dtype,
            )
        )
        self.config.deeploc_label_weights = self.label_weights.detach().cpu().tolist()

    def save_pretrained(self, save_directory, **kwargs):
        # Preserve current ESM keys rather than reversing the old RoPE mapping.
        kwargs.setdefault("save_original_format", False)
        return super().save_pretrained(save_directory, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        final_attention_mask = getattr(
            outputs,
            "final_attention_mask",
            attention_mask,
        )
        if final_attention_mask is None:
            final_attention_mask = torch.ones(
                hidden_states.shape[:2],
                dtype=torch.long,
                device=hidden_states.device,
            )

        logits = self.classifier(
            hidden_states,
            final_attention_mask,
        )
        loss = None
        if labels is not None:
            labels = labels.to(logits.dtype)
            unweighted_bce = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            weighted_bce = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                pos_weight=self.label_weights.to(logits.dtype),
                reduction="none",
            )
            target_probabilities = torch.exp(-unweighted_bce)
            focal_factor = (1.0 - target_probabilities).pow(self.focal_gamma)
            loss = (focal_factor * weighted_bce).mean()

        if not return_dict:
            output = (logits,) + outputs.to_tuple()[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
