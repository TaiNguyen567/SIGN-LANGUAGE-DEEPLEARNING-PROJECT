"""Build dataset splits and metadata manifests for the 3-region VSL dataset."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_metadata() -> None:
    proc_dir = ROOT / "dataset" / "Processed"
    if not proc_dir.is_dir():
        raise FileNotFoundError(f"Processed dataset directory not found: {proc_dir}")

    label_map_file = proc_dir / "label_map.json"
    if not label_map_file.is_file():
        raise FileNotFoundError(f"label_map.json not found: {label_map_file}")

    label_map = json.loads(label_map_file.read_text(encoding="utf-8"))
    print(f"Loaded label_map with {len(label_map)} classes.")

    # Sort classes strictly by class id
    sorted_classes = [item[0] for item in sorted(label_map.items(), key=lambda x: x[1])]

    # 1. Build artifacts/vocab.json
    vocab_tokens = ["<blank>", "<unk>"] + sorted_classes
    vocab_path = ROOT / "artifacts" / "vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    vocab_path.write_text(
        json.dumps({"tokens": vocab_tokens}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Created vocabulary at {vocab_path} ({len(vocab_tokens)} total tokens).")

    # 2. Build metadata CSVs for train, val, test
    metadata_dir = ROOT / "dataset" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    summary_counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        split_dir = proc_dir / split
        if not split_dir.is_dir():
            print(f"Warning: {split_dir} not found, skipping.")
            continue

        rows: list[dict[str, str | int]] = []
        for cls_folder in split_dir.iterdir():
            if not cls_folder.is_dir():
                continue
            cls_name = cls_folder.name
            lbl = label_map.get(cls_name, -1)
            for npz_file in cls_folder.glob("*.npz"):
                rel_path = npz_file.relative_to(ROOT).as_posix()
                rows.append({
                    "feature_path": rel_path,
                    "text": cls_name,
                    "token_text": cls_name,
                    "label": lbl,
                    "language": "vi",
                })

        df = pd.DataFrame(rows)
        # Sort by label and feature path for reproducibility
        df.sort_values(by=["label", "feature_path"], inplace=True)
        out_csv = metadata_dir / f"{split}.csv"
        df.to_csv(out_csv, index=False, encoding="utf-8")
        summary_counts[split] = len(df)
        print(f"Wrote {split}.csv with {len(df)} samples.")

    # 3. Map raw 3-region videos if dataset/Dataset exists
    raw_label_csv = ROOT / "dataset" / "Dataset" / "Labels" / "label.csv"
    raw_video_dir = ROOT / "dataset" / "Dataset" / "Videos"
    if raw_label_csv.is_file() and raw_video_dir.is_dir():
        raw_df = pd.read_csv(raw_label_csv)
        video_rows: list[dict[str, str]] = []
        for _, row in raw_df.iterrows():
            vname = str(row["VIDEO"]).strip()
            vpath = (raw_video_dir / vname).relative_to(ROOT).as_posix()
            stem = vname.replace(".mp4", "")
            if stem.endswith("B"):
                region = "Miền Bắc"
            elif stem.endswith("T"):
                region = "Miền Trung"
            elif stem.endswith("N"):
                region = "Miền Nam"
            else:
                region = "Dùng chung"

            video_rows.append({
                "video_path": vpath,
                "text": str(row["LABEL"]).strip(),
                "region": region,
                "split_group": f"region:{region}",
                "language": "vi",
            })
        video_summary_csv = metadata_dir / "videos_3mien.csv"
        pd.DataFrame(video_rows).to_csv(video_summary_csv, index=False, encoding="utf-8")
        print(f"Wrote {video_summary_csv} with {len(video_rows)} raw 3-region videos.")

    print("Metadata generation complete:", summary_counts)


if __name__ == "__main__":
    build_metadata()
