"""Cache landmarks and write metadata for one class folder of short MP4 clips."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.landmark_extractor import FeatureConfig, LandmarkExtractor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-dir", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    class_dir = Path(args.class_dir)
    if not class_dir.is_absolute():
        class_dir = ROOT / class_dir
    videos = sorted(class_dir.glob("*.mp4"))
    if len(videos) < 3:
        print(f"Need at least 3 videos, found {len(videos)} in {class_dir}", file=sys.stderr)
        return 2

    order = videos[:]
    random.Random(args.seed).shuffle(order)
    n_val = max(1, round(len(order) * 0.15))
    n_test = max(1, round(len(order) * 0.15))
    val_videos = order[:n_val]
    test_videos = order[n_val:n_val + n_test]
    train_videos = order[n_val + n_test:]

    extractor = LandmarkExtractor(FeatureConfig(layout="vsl_201", use_hands=True, use_pose=True, use_face=False))
    rows: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}
    try:
        for split, split_videos in ("train", train_videos), ("val", val_videos), ("test", test_videos):
            out_dir = ROOT / "features" / "xin_loi" / split
            out_dir.mkdir(parents=True, exist_ok=True)
            for video in split_videos:
                sequence = extractor.extract_sequence(video)
                out_path = out_dir / f"{video.stem}.npy"
                np.save(out_path, sequence.astype(np.float32))
                rows[split].append({
                    "feature_path": out_path.relative_to(ROOT).as_posix(),
                    "video_path": video.relative_to(ROOT).as_posix(),
                    "text": args.class_name,
                    "token_text": args.class_name,
                    "language": "vi",
                })
                print(f"[{split}] {video.name} -> {sequence.shape}", flush=True)
    finally:
        extractor.close()

    metadata_dir = ROOT / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        frame = pd.DataFrame(split_rows)
        frame.to_csv(metadata_dir / f"{split}.csv", index=False, encoding="utf-8")
        print(f"wrote {split}.csv ({len(frame)} rows)")

    vocab_path = ROOT / "artifacts" / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps({"tokens": ["<blank>", "<unk>", args.class_name]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {vocab_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
