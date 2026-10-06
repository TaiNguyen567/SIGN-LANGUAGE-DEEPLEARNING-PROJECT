"""Extract cached landmarks and create leakage-resistant metadata splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.preprocessing import split_records
from src.features.landmark_extractor import LandmarkExtractor
from src.utils.config import feature_config_from_dict, load_config, resolve_project_path
from src.utils.logger import configure_logger


def prepare_dataset(
    metadata_path: Path,
    output_dir: Path,
    feature_dir: Path,
    config_path: Path,
    seed: int,
    stratify_by: str | None = None,
) -> dict[str, int]:
    config = load_config(config_path)
    logger = configure_logger("sign_language_ai.prepare", resolve_project_path("logs/app.log"), console=True)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Input metadata not found: {metadata_path}")
    frame = pd.read_csv(metadata_path, keep_default_na=False)
    required = {"video_path", "text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input metadata contains no videos")

    rows = frame.to_dict(orient="records")
    splits = split_records(rows, seed=seed, stratify_key=stratify_by)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    feature_config = feature_config_from_dict(config.get("features", {}))
    extractor_digest = hashlib.sha256()
    for source_path in (
        ROOT / "src/features/landmark_extractor.py",
        ROOT / "src/features/normalization.py",
    ):
        extractor_digest.update(source_path.read_bytes())
    extractor_revision = extractor_digest.hexdigest()[:12]
    logger.info("feature extractor revision=%s", extractor_revision)
    counts: dict[str, int] = {}
    extractor = LandmarkExtractor(feature_config)
    try:
        for split_name, split_rows in splits.items():
            split_features = feature_dir / split_name
            split_features.mkdir(parents=True, exist_ok=True)
            converted: list[dict[str, str]] = []
            skipped_without_landmarks = 0
            for row_number, row in enumerate(split_rows):
                video_path = resolve_project_path(str(row["video_path"]), project_root=ROOT)
                if not video_path.is_file():
                    raise FileNotFoundError(f"Video in {split_name} row {row_number} not found: {video_path}")
                cache_key = hashlib.sha256(
                    f"{video_path.resolve()}|{feature_config}|{extractor_revision}|{video_path.stat().st_size}|{video_path.stat().st_mtime_ns}".encode("utf-8")
                ).hexdigest()[:12]
                stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video_path.stem).strip("_") or "video"
                feature_path = split_features / f"{stem}_{cache_key}.npy"
                if feature_path.is_file():
                    features = np.load(feature_path, allow_pickle=False)
                else:
                    features = extractor.extract_sequence(video_path)

                if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] != feature_config.feature_dim:
                    raise ValueError(f"Unexpected feature shape for {video_path}: {features.shape}")
                if not np.isfinite(features).all():
                    raise ValueError(f"Features contain NaN or infinity for {video_path}")
                if not np.any(features[:, 3::4] > 0.5):
                    skipped_without_landmarks += 1
                    logger.warning("excluding %s row %d: no landmarks detected", split_name, row_number)
                    continue
                if not feature_path.is_file():
                    np.save(feature_path, features, allow_pickle=False)
                record = {
                    "video_path": str(row["video_path"]),
                    "feature_path": feature_path.relative_to(ROOT).as_posix(),
                    "text": str(row["text"]).strip(),
                    "language": str(row.get("language", "vi") or "vi"),
                }
                for optional_column in ("token_text", "split_group"):
                    value = str(row.get(optional_column, "")).strip()
                    if value:
                        record[optional_column] = value
                if str(row.get("subject_id", "")).strip():
                    record["subject_id"] = str(row["subject_id"]).strip()
                converted.append(record)
                if (row_number + 1) % 50 == 0 or row_number + 1 == len(split_rows):
                    logger.info("prepared %s features: %d/%d videos", split_name, row_number + 1, len(split_rows))
            pd.DataFrame(converted).to_csv(output_dir / f"{split_name}.csv", index=False, encoding="utf-8")
            counts[split_name] = len(converted)
            logger.info("prepared %s split with %d videos", split_name, len(converted))
            if skipped_without_landmarks:
                logger.info("excluded %d videos without detectable landmarks from %s", skipped_without_landmarks, split_name)
    finally:
        extractor.close()
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="dataset/metadata/all.csv", help="CSV with video_path,text and optional subject_id")
    parser.add_argument("--output-dir", default="dataset/metadata")
    parser.add_argument("--feature-dir", default="features")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by", help="Optional metadata column used to preserve class proportions")
    args = parser.parse_args()
    try:
        counts = prepare_dataset(
            resolve_project_path(args.metadata, project_root=ROOT),
            resolve_project_path(args.output_dir, project_root=ROOT),
            resolve_project_path(args.feature_dir, project_root=ROOT),
            resolve_project_path(args.config, project_root=ROOT),
            args.seed,
            args.stratify_by,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 2
    print("Prepared video counts:", json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
