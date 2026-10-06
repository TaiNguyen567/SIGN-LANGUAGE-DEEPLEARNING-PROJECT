import csv
import shutil
import zipfile

import pytest

from scripts.import_vsl100 import import_vsl100


def _write_archive(path, entries):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def test_importer_extracts_labeled_train_and_deduplicates_exact_copies(tmp_path):
    archive_path = tmp_path / "source.zip"
    _write_archive(archive_path, [
        ("dataset/train/An ủi/100001.mp4", b"same video"),
        ("dataset/train/An ủi/100001_1.mp4", b"same video"),
        ("dataset/train/Xin lỗi/100002.mp4", b"another video"),
        ("dataset/public_test/100003.mp4", b"unlabeled test"),
    ])
    output_dir = tmp_path / "videos"
    metadata_path = tmp_path / "metadata.csv"
    try:
        result = import_vsl100(
            archive_path,
            output_dir,
            metadata_path,
            expected_sha256=None,
            expected_class_count=2,
        )
        assert result["archive_training_clips"] == 3
        assert result["unique_training_clips"] == 2
        assert result["identical_copies_removed"] == 1
        with metadata_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        assert len(rows) == 2
        assert rows[0]["text"] == "An ủi"
        assert rows[0]["token_text"] == "An_ủi"
        assert rows[0]["split_group"] == "An ủi:100001"
        assert (output_dir / "An ủi" / "100001.mp4").read_bytes() == b"same video"
        assert not (output_dir / "An ủi" / "100001_1.mp4").exists()
    finally:
        for path in (metadata_path, metadata_path.with_suffix(".csv.tmp")):
            path.unlink(missing_ok=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_importer_rejects_conflicting_variants(tmp_path):
    archive_path = tmp_path / "source.zip"
    _write_archive(archive_path, [
        ("dataset/train/Class A/100001.mp4", b"one"),
        ("dataset/train/Class A/100001_1.mp4", b"different"),
    ])

    with pytest.raises(ValueError, match="Conflicting clip variants"):
        import_vsl100(
            archive_path,
            tmp_path / "videos",
            tmp_path / "metadata.csv",
            expected_sha256=None,
            expected_class_count=1,
        )
