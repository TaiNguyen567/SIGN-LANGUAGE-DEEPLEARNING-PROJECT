"""Extract MediaPipe features from recorded raw videos for new classes (e.g. 'thức dậy').

Usage:
  1. Copy your .mp4 videos into dataset/raw_videos/thuc_day/
  2. Run: .venv\\Scripts\\python.exe scripts/extract_new_class.py --class-name "thức dậy" --raw-dir dataset/raw_videos/thuc_day
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.landmark_extractor import FeatureConfig, LandmarkExtractor


def extract_videos_for_class(
    class_name: str,
    raw_dir: Path,
    target_frames: int = 60,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        print(f"[ERROR] Directory not found: {raw_dir}")
        return

    mp4_files = sorted(list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.avi")) + list(raw_dir.glob("*.mov")))
    if not mp4_files:
        print(f"[INFO] No video files found in {raw_dir}.")
        print("Please place your video files (.mp4) in that folder and run this script again.")
        return

    print(f"[INFO] Found {len(mp4_files)} videos in {raw_dir}. Initializing MediaPipe extractor (vsl_201 layout)...")

    config = FeatureConfig(
        layout="vsl_201",
        use_hands=True,
        use_pose=True,
        use_face=False,
    )
    extractor = LandmarkExtractor(config)

    proc_dir = ROOT / "dataset" / "Processed"
    for split in ("train", "val", "test"):
        (proc_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    # Determine split indices
    n = len(mp4_files)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio)) if n > 2 else 0

    saved_counts = {"train": 0, "val": 0, "test": 0}

    for idx, video_file in enumerate(mp4_files):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_val:
            split = "val"
        else:
            split = "test"

        out_name = f"{idx}.npz"
        out_path = proc_dir / split / class_name / out_name

        try:
            seq = extractor.extract_sequence(video_file)
            if len(seq) == 0:
                print(f"[WARN] Skipped {video_file.name} (0 frames extracted)")
                continue

            # Resample / interpolate to target_frames
            if len(seq) != target_frames:
                indices = np.linspace(0, len(seq) - 1, target_frames).round().astype(int)
                seq = seq[indices]

            np.savez_compressed(out_path, sequence=seq.astype(np.float64), label=1)
            saved_counts[split] += 1
            print(f"[{split.upper()}] Processed {video_file.name} -> {out_name} (shape {seq.shape})")
        except Exception as exc:
            print(f"[ERROR] Failed processing {video_file.name}: {exc}")

    extractor.close()
    print(f"\n[SUMMARY] Saved: train={saved_counts['train']}, val={saved_counts['val']}, test={saved_counts['test']}")

    # Refresh metadata CSVs
    print("[INFO] Updating dataset/metadata manifests...")
    meta_dir = ROOT / "dataset" / "metadata"
    vocab_file = ROOT / "artifacts" / "vocab.json"
    import json
    vocab = json.loads(vocab_file.read_text(encoding="utf-8"))
    tokens = vocab.get("tokens", [])
    label_id = tokens.index(class_name) if class_name in tokens else len(tokens)

    for split in ("train", "val", "test"):
        csv_path = meta_dir / f"{split}.csv"
        df = pd.read_csv(csv_path) if csv_path.is_file() else pd.DataFrame()

        # Remove existing rows for this class to avoid duplicates
        if not df.empty and "text" in df.columns:
            df = df[df["text"] != class_name]

        # Add newly extracted files
        cls_split_dir = proc_dir / split / class_name
        new_rows = []
        for npz in sorted(cls_split_dir.glob("*.npz")):
            rel_path = npz.relative_to(ROOT).as_posix()
            new_rows.append({
                "feature_path": rel_path,
                "text": class_name,
                "token_text": class_name,
                "label": label_id,
                "language": "vi",
            })

        if new_rows:
            new_df = pd.DataFrame(new_rows)
            df = pd.concat([df, new_df], ignore_index=True)

        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[OK] Updated {split}.csv ({len(df)} total samples)")

    print(f"\n[DONE] Successfully extracted and registered class '{class_name}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", default="thức dậy", help="Name of the gesture class")
    parser.add_argument("--raw-dir", default="dataset/raw_videos/thuc_day", help="Directory containing raw videos")
    args = parser.parse_args()

    extract_videos_for_class(args.class_name, ROOT / args.raw_dir)


if __name__ == "__main__":
    main()
