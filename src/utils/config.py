"""YAML configuration loading with path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.features.landmark_extractor import FeatureConfig


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def resolve_project_path(value: str | Path, *, project_root: str | Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    return (root / path).resolve()


def feature_config_from_dict(values: dict[str, Any]) -> FeatureConfig:
    settings = dict(values)
    if "face_landmark_indices" in settings:
        settings["face_landmark_indices"] = tuple(settings["face_landmark_indices"])
    allowed = {"use_hands", "use_pose", "use_face", "face_landmark_indices", "min_visibility", "model_complexity"}
    return FeatureConfig(**{key: value for key, value in settings.items() if key in allowed})
