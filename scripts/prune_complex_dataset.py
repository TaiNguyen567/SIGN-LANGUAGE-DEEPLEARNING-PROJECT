"""Safely prune complex / unneeded dataset classes to free disk space while preserving simple target classes."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.filter_basic_dataset import PRESETS


def get_allowed_classes(preset: str) -> set[str]:
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESETS.keys())}")
    return set(PRESETS[preset])


def prune_processed_dataset(
    allowed_classes: set[str],
    dry_run: bool = True,
    remove_raw_videos: bool = False,
) -> None:
    proc_dir = ROOT / "dataset" / "Processed"
    if not proc_dir.is_dir():
        print(f"Directory {proc_dir} does not exist.")
        return

    splits = ["train", "val", "test"]
    deleted_folders = 0
    retained_folders = 0
    bytes_to_free = 0

    print(f"Target whitelist classes count: {len(allowed_classes)}")
    print("Scanning dataset/Processed...")

    folders_to_delete: list[Path] = []

    for split in splits:
        split_path = proc_dir / split
        if not split_path.is_dir():
            continue
        for class_dir in split_path.iterdir():
            if not class_dir.is_dir():
                continue
            if class_dir.name not in allowed_classes:
                folders_to_delete.append(class_dir)
                for f in class_dir.glob("*"):
                    if f.is_file():
                        bytes_to_free += f.stat().st_size
                deleted_folders += 1
            else:
                retained_folders += 1

    gb_to_free = bytes_to_free / (1024 ** 3)
    mb_to_free = bytes_to_free / (1024 ** 2)

    raw_video_dir = ROOT / "dataset" / "Dataset"
    raw_video_bytes = 0
    if remove_raw_videos and raw_video_dir.is_dir():
        for f in raw_video_dir.rglob("*"):
            if f.is_file():
                raw_video_bytes += f.stat().st_size

    print("\n--- PRUNE SUMMARY ---")
    print(f"Retained class directories: {retained_folders}")
    print(f"Directories to delete:     {deleted_folders}")
    print(f"Processed space to free:   {mb_to_free:.2f} MB ({gb_to_free:.2f} GB)")
    if remove_raw_videos:
        print(f"Raw videos space to free:  {raw_video_bytes / (1024**3):.2f} GB")

    if dry_run:
        print("\n[DRY RUN] No files were deleted.")
        print("To permanently delete, re-run with --confirm.")
        return

    print("\n[DELETING] Removing unselected class directories...")
    for folder in folders_to_delete:
        shutil.rmtree(folder, ignore_errors=True)

    if remove_raw_videos and raw_video_dir.is_dir():
        print(f"[DELETING] Removing raw dataset directory {raw_video_dir}...")
        shutil.rmtree(raw_video_dir, ignore_errors=True)

    print("Pruning completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["basic", "standard", "extended"],
        default="basic",
        help="Target whitelist to keep: basic (47 classes), standard (116 classes), extended (169 classes). Default: basic",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm permanent deletion of complex class directories from disk.",
    )
    parser.add_argument(
        "--remove-raw",
        action="store_true",
        help="Also remove dataset/Dataset raw videos directory (frees ~2.7 GB).",
    )
    args = parser.parse_args()

    allowed = get_allowed_classes(args.preset)
    prune_processed_dataset(
        allowed_classes=allowed,
        dry_run=not args.confirm,
        remove_raw_videos=args.remove_raw,
    )


if __name__ == "__main__":
    main()
