"""Load checkpoints only after validating their model and vocabulary contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from src.data.tokenizer import SignTokenizer
from src.models.transformer import SignLanguageTransformer


def load_model_checkpoint(
    path: str | Path,
    *,
    device: torch.device,
    expected_feature_dim: int | None = None,
    expected_model_config: dict[str, Any] | None = None,
) -> tuple[SignLanguageTransformer, SignTokenizer, dict[str, Any]]:
    """Load a trusted local checkpoint and reject incompatible project settings."""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. Train a model first with `python train.py`."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    required = {"model_state_dict", "model_config", "vocab", "feature_dim", "config"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {', '.join(sorted(missing))}")

    model_config = dict(checkpoint["model_config"])
    vocabulary = checkpoint["vocab"]
    tokenizer = SignTokenizer(vocabulary)
    feature_dim = int(checkpoint["feature_dim"])
    if int(model_config.get("input_dim", -1)) != feature_dim:
        raise ValueError("Checkpoint feature_dim does not match the saved model architecture")
    if int(model_config.get("num_classes", -1)) != tokenizer.vocabulary_size:
        raise ValueError("Checkpoint vocabulary size does not match the model class count")
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError(
            f"Feature dimension mismatch: checkpoint expects {feature_dim}, current config produces {expected_feature_dim}"
        )
    if expected_model_config is not None:
        config_keys = {
            "d_model": "d_model",
            "nhead": "nhead",
            "num_layers": "num_layers",
            "dim_feedforward": "dim_feedforward",
            "dropout": "dropout",
        }
        for config_key, checkpoint_key in config_keys.items():
            if config_key in expected_model_config and expected_model_config[config_key] is not None:
                configured = expected_model_config[config_key]
                saved = model_config.get(checkpoint_key)
                if float(configured) != float(saved):
                    raise ValueError(
                        f"Model configuration mismatch for {config_key}: checkpoint={saved}, current={configured}"
                    )

    model = SignLanguageTransformer(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, tokenizer, checkpoint
