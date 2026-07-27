"""Sequence pooling operations used between ESM transformer layers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MaskedMeanPool1d(nn.Module):
    """Average residue pairs while preserving ``<cls>`` and ``<eos>``.

    This module has no trainable parameters. It assumes right-padded ESM
    sequences with ``<cls>`` first and ``<eos>`` as the last valid token.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, L, D]")
        if attention_mask.ndim != 2:
            raise ValueError("attention_mask must have shape [B, L]")

        batch_size, sequence_length, hidden_size = hidden_states.shape
        if sequence_length < 3:
            raise ValueError("a sequence must contain <cls>, a residue and <eos>")
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError("attention_mask and hidden_states are incompatible")

        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.shape[0] == 1:
            position_ids = position_ids.expand(batch_size, -1)
        if position_ids.shape != (batch_size, sequence_length):
            raise ValueError("position_ids and hidden_states are incompatible")

        valid_lengths = attention_mask.to(torch.long).sum(dim=1)
        eos_indices = valid_lengths - 1

        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        eos_states = hidden_states[batch_indices, eos_indices]
        eos_positions = position_ids[batch_indices, eos_indices]

        # The longest sequence has <eos> in the last column. Removing the
        # first and last columns therefore leaves all possible residue slots.
        residue_states = hidden_states[:, 1:-1]
        residue_positions = position_ids[:, 1:-1].to(torch.float32)
        residue_columns = torch.arange(
            1,
            sequence_length - 1,
            device=hidden_states.device,
        )
        residue_mask = attention_mask[:, 1:-1].bool() & (
            residue_columns.unsqueeze(0) < eos_indices.unsqueeze(1)
        )

        # Add one empty slot when needed so the residue dimension is even.
        pad_slots = residue_states.shape[1] % 2
        if pad_slots:
            residue_states = F.pad(residue_states, (0, 0, 0, 1))
            residue_positions = F.pad(residue_positions, (0, 1))
            residue_mask = F.pad(residue_mask, (0, 1))

        pooled_slots = residue_states.shape[1] // 2
        grouped_states = residue_states.reshape(
            batch_size,
            pooled_slots,
            2,
            hidden_size,
        )
        grouped_positions = residue_positions.reshape(
            batch_size,
            pooled_slots,
            2,
        )
        grouped_mask = residue_mask.reshape(batch_size, pooled_slots, 2)

        counts = grouped_mask.sum(dim=2).clamp_min(1)
        state_weights = grouped_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled_states = (
            (grouped_states * state_weights).sum(dim=2)
            / counts.unsqueeze(-1).to(hidden_states.dtype)
        )
        pooled_positions = (
            grouped_positions * grouped_mask.to(grouped_positions.dtype)
        ).sum(dim=2) / counts.to(grouped_positions.dtype)
        pooled_mask = grouped_mask.any(dim=2)

        output_length = pooled_slots + 2
        output_states = hidden_states.new_zeros(
            (batch_size, output_length, hidden_size)
        )
        output_states[:, 0] = hidden_states[:, 0]
        output_states[:, 1 : pooled_slots + 1] = pooled_states

        output_mask = attention_mask.new_zeros(
            (batch_size, output_length)
        )
        output_mask[:, 0] = attention_mask[:, 0]
        output_mask[:, 1 : pooled_slots + 1] = pooled_mask.to(
            attention_mask.dtype
        )

        output_positions = pooled_positions.new_zeros(
            (batch_size, output_length)
        )
        output_positions[:, 0] = position_ids[:, 0]
        output_positions[:, 1 : pooled_slots + 1] = pooled_positions

        residue_lengths = (valid_lengths - 2).clamp_min(0)
        pooled_lengths = (residue_lengths + 1) // 2
        output_eos_indices = pooled_lengths + 1

        output_states[batch_indices, output_eos_indices] = eos_states
        output_mask[batch_indices, output_eos_indices] = 1
        output_positions[batch_indices, output_eos_indices] = (
            eos_positions.to(output_positions.dtype)
        )

        return output_states, output_mask, output_positions
