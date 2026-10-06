"""CTC loss setup for transcript-level supervision."""

from __future__ import annotations

import torch
from torch import nn


class CTCLoss(nn.Module):
    """Thin validation wrapper around PyTorch's transcript-level CTC loss."""

    def __init__(self, blank_id: int = 0, zero_infinity: bool = True) -> None:
        super().__init__()
        self.blank_id = blank_id
        self.loss = nn.CTCLoss(blank=blank_id, zero_infinity=zero_infinity)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        if log_probs.ndim != 3:
            raise ValueError("log_probs must have shape [time, batch, classes]")
        if targets.ndim != 1:
            raise ValueError("targets must be a flattened 1D tensor")
        batch_size = log_probs.shape[1]
        if input_lengths.numel() != batch_size or target_lengths.numel() != batch_size:
            raise ValueError("input_lengths and target_lengths must match batch size")
        return self.loss(
            log_probs.float(),
            targets.to(device=log_probs.device, dtype=torch.long),
            input_lengths.detach().to(device="cpu", dtype=torch.long),
            target_lengths.detach().to(device="cpu", dtype=torch.long),
        )
