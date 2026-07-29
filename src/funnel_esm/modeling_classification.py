"""Shared masked-attention classifier for baseline and Funnel ESM."""

from __future__ import annotations

import torch
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.esm.modeling_esm import (
    EsmForSequenceClassification,
)

from funnel_esm.pooling import MaskedAttentionClassificationHead


class MaskedAttentionEsmForSequenceClassification(
    EsmForSequenceClassification
):
    """ESM classifier with DeepLoc-style pooling and weighted focal loss."""

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
            raise ValueError(
                "label_weights must contain one value per label"
            )
        self.label_weights.copy_(
            label_weights.to(
                device=self.label_weights.device,
                dtype=self.label_weights.dtype,
            )
        )
        self.config.deeploc_label_weights = (
            self.label_weights.detach().cpu().tolist()
        )

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
        return_dict = (
            return_dict
            if return_dict is not None
            else self.config.use_return_dict
        )

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
            binary_cross_entropy = F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
            probabilities = torch.sigmoid(logits)
            target_probabilities = torch.where(
                labels.bool(),
                probabilities,
                1 - probabilities,
            )
            focal_factor = (
                1 - target_probabilities
            ).pow(self.focal_gamma)
            loss = (
                binary_cross_entropy
                * focal_factor
                * self.label_weights.to(logits.dtype).unsqueeze(0)
            ).mean()

        if not return_dict:
            output = (logits,) + outputs.to_tuple()[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
