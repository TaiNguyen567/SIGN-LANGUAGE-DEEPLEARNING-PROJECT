"""Leakage-resistant video/subject splits and metadata validation."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterable


def split_records(
    records: Iterable[dict[str, Any]],
    *,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    subject_key: str = "subject_id",
    group_key: str = "split_group",
    video_key: str = "video_path",
    stratify_key: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Split whole subjects/groups, optionally preserving label proportions."""
    rows = [dict(record) for record in records]
    if not rows:
        raise ValueError("Cannot split an empty dataset")
    if len(fractions) != 3 or any(value <= 0 for value in fractions):
        raise ValueError("fractions must contain three positive values")
    total = sum(fractions)
    fractions = tuple(value / total for value in fractions)
    if any(video_key not in row or not str(row[video_key]).strip() for row in rows):
        raise ValueError(f"Every record must include a non-empty {video_key!r}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject = str(row.get(subject_key, "")).strip()
        split_group = str(row.get(group_key, "")).strip()
        group = (
            f"subject:{subject}" if subject
            else f"group:{split_group}" if split_group
            else f"video:{row[video_key]}"
        )
        groups[group].append(row)

    result: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    if stratify_key:
        strata: dict[str, list[str]] = defaultdict(list)
        for group_key_value, group_rows in groups.items():
            labels = {str(row.get(stratify_key, "")).strip() for row in group_rows}
            if "" in labels:
                raise ValueError(f"Every record must include a non-empty {stratify_key!r}")
            if len(labels) != 1:
                raise ValueError(f"Split group {group_key_value!r} has multiple {stratify_key!r} values")
            strata[next(iter(labels))].append(group_key_value)
        for label in sorted(strata):
            keys = strata[label]
            rng.shuffle(keys)
            cursor = 0
            for name, count in zip(result, _split_counts(len(keys), fractions)):
                for key in keys[cursor:cursor + count]:
                    result[name].extend(groups[key])
                cursor += count
        for split_rows in result.values():
            rng.shuffle(split_rows)
    else:
        keys = list(groups)
        rng.shuffle(keys)
        cursor = 0
        for name, count in zip(result, _split_counts(len(keys), fractions)):
            for key in keys[cursor:cursor + count]:
                result[name].extend(groups[key])
            cursor += count
    validate_splits(result, subject_key=subject_key, group_key=group_key, video_key=video_key)
    return result


def validate_splits(
    splits: dict[str, Iterable[dict[str, Any]]],
    *,
    subject_key: str = "subject_id",
    group_key: str = "split_group",
    video_key: str = "video_path",
) -> None:
    """Raise when a video, split group, or known subject leaks across splits."""
    video_owner: dict[str, str] = {}
    subject_owner: dict[str, str] = {}
    group_owner: dict[str, str] = {}
    for split_name, records in splits.items():
        for row in records:
            video = str(row[video_key]).strip()
            previous = video_owner.setdefault(video, split_name)
            if previous != split_name:
                raise ValueError(f"Video leakage: {video!r} occurs in {previous} and {split_name}")
            subject = str(row.get(subject_key, "")).strip()
            if subject:
                previous = subject_owner.setdefault(subject, split_name)
                if previous != split_name:
                    raise ValueError(f"Subject leakage: {subject!r} occurs in {previous} and {split_name}")
            split_group = str(row.get(group_key, "")).strip()
            if split_group:
                previous = group_owner.setdefault(split_group, split_name)
                if previous != split_name:
                    raise ValueError(f"Group leakage: {split_group!r} occurs in {previous} and {split_name}")


def _split_counts(group_count: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if group_count == 1:
        return (1, 0, 0)
    if group_count == 2:
        return (1, 0, 1)
    raw = [group_count * value for value in fractions]
    counts = [int(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[:group_count - sum(counts)]:
        counts[index] += 1
    for index in range(3):
        if counts[index] == 0:
            donor = max(range(3), key=lambda i: counts[i])
            counts[donor] -= 1
            counts[index] = 1
    return tuple(counts)
