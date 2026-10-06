"""Transcript dataset backed by cached landmarks or raw video files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.tokenizer import SignTokenizer
from src.data.augmentation import LandmarkAugmenter
from src.features.landmark_extractor import FeatureConfig, LandmarkExtractor


class SignLanguageDataset(Dataset[dict[str, Any]]):
    """Load one video feature sequence and its transcript token IDs per item."""

    REQUIRED_COLUMNS = ("video_path", "text")

    def __init__(
        self,
        metadata_path: str | Path,
        tokenizer: SignTokenizer,
        *,
        project_root: str | Path | None = None,
        feature_config: FeatureConfig | None = None,
        max_frames: int | None = None,
        augmenter: LandmarkAugmenter | None = None,
    ) -> None:
        self.metadata_path = Path(metadata_path).resolve()
        self.project_root = Path(project_root).resolve() if project_root else self.metadata_path.parents[2]
        self.tokenizer = tokenizer
        self.feature_config = feature_config or FeatureConfig()
        self.max_frames = max_frames
        self.augmenter = augmenter
        self._extractor: LandmarkExtractor | None = None

        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"Metadata CSV not found: {self.metadata_path}")
        self.metadata = pd.read_csv(self.metadata_path, keep_default_na=False)
        missing = [column for column in self.REQUIRED_COLUMNS if column not in self.metadata.columns]
        if missing:
            raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
        if self.metadata.empty:
            raise ValueError(f"Dataset metadata contains no samples: {self.metadata_path}")
        if (self.metadata["text"].astype(str).str.strip() == "").any():
            raise ValueError("Every video must have a non-empty transcript")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.metadata.iloc[index]
        features = self._load_features(row)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError(f"Feature sequence must have shape [T, F] with T > 0 (row {index})")
        if not np.isfinite(features).all():
            raise ValueError(f"Feature sequence contains NaN or infinity (row {index})")
        if self.max_frames and features.shape[0] > self.max_frames:
            indices = np.linspace(0, features.shape[0] - 1, self.max_frames).round().astype(int)
            features = features[indices]
        if self.augmenter is not None:
            features = self.augmenter(features)

        text = str(row["text"])
        token_text = str(row.get("token_text", "")).strip() or text
        target = self.tokenizer.encode(token_text)
        if not target:
            raise ValueError(f"Transcript tokenized to an empty sequence (row {index})")
        return {
            "features": torch.from_numpy(np.asarray(features, dtype=np.float32)),
            "target": torch.tensor(target, dtype=torch.long),
            "text": text,
            "video_path": str(row["video_path"]),
        }

    def close(self) -> None:
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None

    def _load_features(self, row: pd.Series) -> np.ndarray:
        feature_value = str(row.get("feature_path", "")).strip()
        if feature_value:
            feature_path = self._resolve_path(feature_value)
            if not feature_path.is_file():
                raise FileNotFoundError(f"Cached feature file not found: {feature_path}")
            return np.load(feature_path, allow_pickle=False).astype(np.float32, copy=False)

        video_path = self._resolve_path(str(row["video_path"]))
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if self._extractor is None:
            self._extractor = LandmarkExtractor(self.feature_config)
        return self._extractor.extract_sequence(video_path)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.project_root / path).resolve()


def ctc_collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad input sequences and flatten CTC targets for ``torch.nn.CTCLoss``."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    feature_dims = {sample["features"].shape[1] for sample in samples}
    if len(feature_dims) != 1:
        raise ValueError("All feature sequences in a batch must have the same feature dimension")

    input_lengths = torch.tensor([sample["features"].shape[0] for sample in samples], dtype=torch.long)
    target_lengths = torch.tensor([sample["target"].numel() for sample in samples], dtype=torch.long)
    max_frames = int(input_lengths.max().item())
    batch_size = len(samples)
    feature_dim = next(iter(feature_dims))
    features = torch.zeros(batch_size, max_frames, feature_dim, dtype=torch.float32)
    for index, sample in enumerate(samples):
        features[index, :sample["features"].shape[0]] = sample["features"]

    targets = torch.cat([sample["target"] for sample in samples])
    return {
        "features": features,
        "input_lengths": input_lengths,
        "targets": targets,
        "target_lengths": target_lengths,
        "texts": [sample["text"] for sample in samples],
        "video_paths": [sample["video_path"] for sample in samples],
    }
