import pytest

from src.data.preprocessing import split_records, validate_splits


def test_stratified_split_keeps_classes_and_groups_together():
    records = [
        {
            "video_path": f"{label}-{group}-{copy}.mp4",
            "text": label,
            "split_group": f"{label}-{group}",
        }
        for label, count in (("A", 5), ("B", 7))
        for group in range(count)
        for copy in range(2)
    ]

    splits = split_records(records, seed=13, stratify_key="text")

    validate_splits(splits)
    for split in splits.values():
        assert all(sum(row["text"] == label for row in split) % 2 == 0 for label in ("A", "B"))
    for label in ("A", "B"):
        assert all(any(row["text"] == label for row in splits[name]) for name in ("train", "val", "test"))
    assert splits == split_records(records, seed=13, stratify_key="text")


def test_stratified_split_rejects_mixed_labels_in_one_group():
    records = [
        {"video_path": "one.mp4", "split_group": "same", "text": "A"},
        {"video_path": "two.mp4", "split_group": "same", "text": "B"},
    ]

    with pytest.raises(ValueError, match="multiple 'text' values"):
        split_records(records, stratify_key="text")
