"""Stabilize overlapping CTC window hypotheses for continuous display."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence


def remove_duplicate_tokens(tokens: Iterable[str]) -> list[str]:
    result: list[str] = []
    for token in tokens:
        if not result or token != result[-1]:
            result.append(token)
    return result


def merge_predictions(existing: Sequence[str], prediction: Sequence[str], *, max_overlap: int = 32) -> list[str]:
    """Append only the part of a new window hypothesis not already in the tail."""
    old = list(existing)
    new = list(prediction)
    if not new:
        return old
    limit = min(len(old), len(new), max_overlap)
    overlap = 0
    for size in range(limit, 0, -1):
        if old[-size:] == new[:size]:
            overlap = size
            break
    return old + new[overlap:]


def confidence_filter(tokens: Sequence[str], confidence: float, threshold: float) -> list[str]:
    if confidence < threshold:
        return []
    return list(tokens)


def temporal_smoothing(
    predictions: Sequence[Sequence[str]],
    *,
    min_votes: int = 2,
    window_size: int = 3,
) -> list[str]:
    """Keep tokens that recur at the same transcript position in recent windows."""
    recent = [list(value) for value in predictions[-window_size:]]
    if not recent or min_votes < 1:
        return []
    stable: list[str] = []
    max_length = max(len(value) for value in recent)
    for position in range(max_length):
        votes = [value[position] for value in recent if len(value) > position]
        if not votes:
            break
        counts = Counter(votes)
        last_seen = {token: len(votes) - 1 - votes[::-1].index(token) for token in counts}
        winner = max(counts, key=lambda token: (counts[token], last_seen[token]))
        count = counts[winner]
        if count < min_votes:
            break
        stable.append(winner)
    return stable
