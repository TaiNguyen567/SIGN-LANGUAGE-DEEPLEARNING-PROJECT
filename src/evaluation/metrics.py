"""Character and word error metrics for sequence recognition."""

from __future__ import annotations

import unicodedata
from typing import Iterable


def edit_distance(reference: str, hypothesis: str) -> int:
    """Compute Levenshtein distance using O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_char in enumerate(reference, start=1):
        current = [row]
        for column, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(
                current[column - 1] + 1,
                previous[column] + 1,
                previous[column - 1] + (ref_char != hyp_char),
            ))
        previous = current
    return previous[-1]


def character_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    pairs = _normalized_pairs(references, hypotheses)
    errors = sum(edit_distance(reference, hypothesis) for reference, hypothesis in pairs)
    total = sum(len(reference) for reference, _ in pairs)
    return errors / max(total, 1)


def word_error_rate(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    pairs = _normalized_pairs(references, hypotheses)
    errors = sum(edit_distance(reference.split(), hypothesis.split()) for reference, hypothesis in pairs)
    total = sum(len(reference.split()) for reference, _ in pairs)
    return errors / max(total, 1)


def sentence_accuracy(references: Iterable[str], hypotheses: Iterable[str]) -> float:
    pairs = _normalized_pairs(references, hypotheses)
    if not pairs:
        return 0.0
    return sum(reference == hypothesis for reference, hypothesis in pairs) / len(pairs)


def _normalized_pairs(references: Iterable[str], hypotheses: Iterable[str]) -> list[tuple[str, str]]:
    refs = list(references)
    hyps = list(hypotheses)
    if len(refs) != len(hyps):
        raise ValueError("references and hypotheses must have the same length")
    normalize = lambda text: unicodedata.normalize("NFC", str(text)).strip().upper()
    return [(normalize(ref), normalize(hyp)) for ref, hyp in zip(refs, hyps)]
