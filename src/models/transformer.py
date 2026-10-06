"""Temporal Transformer encoder with a frame-wise CTC classification head."""

from __future__ import annotations

import torch
from torch import nn

from src.models.positional_encoding import PositionalEncoding


class SignLanguageTransformer(nn.Module):
    """Map normalized landmark sequences ``[B, T, F]`` to CTC log-probabilities."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 2048,
    ) -> None:
        super().__init__()
        if input_dim < 1 or num_classes < 2:
            raise ValueError("input_dim must be positive and num_classes must include blank plus one token")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if num_layers < 1 or dim_feedforward < 1:
            raise ValueError("num_layers and dim_feedforward must be positive")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.projection = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model))
        self.position = PositionalEncoding(d_model, dropout=dropout, max_len=max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, features: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return log probabilities with shape ``[T, B, C]`` for CTCLoss."""
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, time, feature_dim]")
        batch_size, time_steps, feature_dim = features.shape
        if feature_dim != self.input_dim:
            raise ValueError(f"Expected feature dimension {self.input_dim}, received {feature_dim}")
        if time_steps < 1:
            raise ValueError("Input sequence must contain at least one frame")

        padding_mask = None
        if lengths is not None:
            lengths = torch.as_tensor(lengths, dtype=torch.long, device=features.device).reshape(-1)
            if lengths.numel() != batch_size:
                raise ValueError("lengths must contain one value per batch item")
            if torch.any(lengths < 1) or torch.any(lengths > time_steps):
                raise ValueError("Every sequence length must be between 1 and padded time_steps")
            padding_mask = torch.arange(time_steps, device=features.device)[None, :] >= lengths[:, None]

        hidden = self.position(self.projection(features))
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.classifier(hidden).log_softmax(dim=-1).transpose(0, 1).contiguous()
