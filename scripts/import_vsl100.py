"""Safely import the VSL 100-class training videos from the pinned source archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tokenizer import SignTokenizer

EXPECTED_SHA256 = "cdfec1080e8b75cf10a972a4d7cf54fcfa4bf1b874002d28eae8405188a2a308"
SOURCE_URL = "https://huggingface.co/datasets/star092304/ViSignLanguage-Video"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
VIDEO_NAME = re.compile(r"^(?P<recording>[0-9]+)(?:_[0-9]+)?\.mp4$", re.IGNORECASE)


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _entry_digest(archive: zipfile.ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_vsl100(
    archive_path: Path,
    video_dir: Path,
    metadata_path: Path,
    *,
    expected_sha256: str | None = EXPECTED_SHA256,
    expected_class_count: int | None = 100,
) -> dict[str, Any]:
    """Extract labeled train clips, collapse identical copies, and write metadata."""
    archive_path = archive_path.resolve()
    video_dir = video_dir.resolve()
    metadata_path = metadata_path.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if _sha256(archive_path) != expected_sha256 and expected_sha256 is not None:
        raise ValueError("Dataset archive SHA-256 does not match the pinned release")

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            member = info.filename
            if not member.startswith("dataset/train/") or not member.lower().endswith(".mp4"):
                continue
            parts = PurePosixPath(member).parts
            if len(parts) != 4 or parts[:2] != ("dataset", "train") or info.is_dir():
                raise ValueError(f"Unexpected training archive path: {member!r}")
            label = unicodedata.normalize("NFC", parts[2]).strip()
            filename = parts[3]
            match = VIDEO_NAME.fullmatch(filename)
            if not label or match is None:
                raise ValueError(f"Unexpected class or clip filename: {member!r}")
            group_id = f"{label}:{match.group('recording')}"
            grouped[(label, group_id)].append(member)

        labels = {label for label, _ in grouped}
        if expected_class_count is not None and len(labels) != expected_class_count:
            raise ValueError(f"Expected {expected_class_count} classes, found {len(labels)}")
        if not grouped:
            raise ValueError("Archive contains no labeled training videos")

        rows: list[dict[str, str]] = []
        duplicates_removed = 0
        for (label, group_id), members in sorted(grouped.items()):
            members.sort(key=lambda member: ("_" in PurePosixPath(member).name, member))
            digests = {_entry_digest(archive, member) for member in members}
            if len(digests) != 1:
                raise ValueError(f"Conflicting clip variants found in group {group_id!r}")
            duplicates_removed += len(members) - 1
            selected = members[0]
            filename = PurePosixPath(selected).name
            target = (video_dir / label / filename).resolve()
            try:
                target.relative_to(video_dir)
            except ValueError as exc:
                raise ValueError(f"Unsafe output path for archive member: {selected!r}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if _sha256(target) not in digests:
                    raise FileExistsError(f"Existing clip differs from the source archive: {target}")
            else:
                try:
                    with archive.open(selected) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination, length=1024 * 1024)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
            display_label = " ".join(label.split())
            token_text = "_".join(display_label.split())
            if len(SignTokenizer.tokenize(token_text)) != 1:
                raise ValueError(f"Class label cannot be represented as one CTC token: {label!r}")
            try:
                video_value = target.relative_to(ROOT).as_posix()
            except ValueError:
                video_value = str(target)
            rows.append({
                "video_path": video_value,
                "text": display_label,
                "token_text": token_text,
                "language": "vi",
                "split_group": group_id,
            })

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temp_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    fields = ("video_path", "text", "token_text", "language", "split_group")
    with temp_metadata.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp_metadata.replace(metadata_path)

    manifest = {
        "dataset": "AI Challenge CV Vietnamese Sign Language, 100 isolated classes",
        "source": SOURCE_URL,
        "license": "CC BY 4.0 as declared by the dataset card",
        "license_url": LICENSE_URL,
        "archive_sha256": _sha256(archive_path),
        "archive_training_clips": sum(map(len, grouped.values())),
        "unique_training_clips": len(rows),
        "identical_copies_removed": duplicates_removed,
        "classes": len(labels),
        "metadata": metadata_path.relative_to(ROOT).as_posix() if metadata_path.is_relative_to(ROOT) else str(metadata_path),
        "excluded_archive_splits": ["public_test", "private_test"],
    }
    manifest_path = archive_path.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="dataset/raw/vsl100/dataset.zip")
    parser.add_argument("--video-dir", default="dataset/videos/vsl100")
    parser.add_argument("--metadata", default="dataset/metadata/vsl100.csv")
    args = parser.parse_args()
    try:
        summary = import_vsl100(
            _project_path(args.archive),
            _project_path(args.video_dir),
            _project_path(args.metadata),
        )
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"VSL100 import failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
