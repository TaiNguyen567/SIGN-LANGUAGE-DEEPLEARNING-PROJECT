"""Sinusoidal position encoding for batch-first temporal features."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Add deterministic temporal position information to projected features."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2048) -> None:
        super().__init__()
        if d_model < 1 or max_len < 1:
            raise ValueError("d_model and max_len must be positive")
        self.dropout = nn.Dropout(dropout)
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        encoding[:, 1::2] = torch.cos(positions * div_term[:encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, channels]")
        if features.shape[1] > self.encoding.shape[1]:
            raise ValueError(f"Sequence length {features.shape[1]} exceeds positional limit {self.encoding.shape[1]}")
        return self.dropout(features + self.encoding[:, :features.shape[1]].to(dtype=features.dtype))
