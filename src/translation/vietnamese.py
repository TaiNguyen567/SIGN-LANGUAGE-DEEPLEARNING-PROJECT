"""Conservative rule-based formatting for recognized Vietnamese sign tokens."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from functools import lru_cache


class VietnamesePostProcessor:
    """Format glosses and apply configurable phrase-order rules."""

    DEFAULT_REORDERING_RULES: dict[tuple[str, ...], tuple[str, ...]] = {}

    def __init__(
        self,
        reordering_rules: Mapping[tuple[str, ...], tuple[str, ...]] | None = None,
    ) -> None:
        rules = self.DEFAULT_REORDERING_RULES if reordering_rules is None else reordering_rules
        self.reordering_rules = {
            self._normalize_rule_phrase(source): self._normalize_rule_phrase(target)
            for source, target in rules.items()
        }
        if any(not source or not target for source, target in self.reordering_rules.items()):
            raise ValueError("Reordering rules must have non-empty source and target phrases")
        self._ordered_rules = sorted(self.reordering_rules.items(), key=lambda rule: len(rule[0]), reverse=True)

    @classmethod
    def from_config(cls, rules: Mapping[str, str] | None) -> "VietnamesePostProcessor":
        if rules is None:
            return cls()
        parsed = {
            tuple(source.split()): tuple(target.split())
            for source, target in rules.items()
        }
        return cls(parsed)

    def translate(self, tokens: str | Sequence[str]) -> str:
        raw = tokens if isinstance(tokens, str) else " ".join(tokens)
        normalized = unicodedata.normalize("NFC", raw.replace("_", " ")).upper().strip()
        normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        if not normalized:
            return ""

        words = normalized.split()
        words = self._reorder(words)
        if len(words) >= 3 and words[0:2] == ["TÔI", "TÊN"]:
            words = ["TÔI", "TÊN", "LÀ", *words[2:]]
        sentence = " ".join(words)
        is_name_phrase = len(words) >= 4 and words[:3] == ["TÔI", "TÊN", "LÀ"]
        sentence = sentence[:1].upper() + sentence[1:].lower()
        if is_name_phrase:
            parts = sentence.split()
            parts[3] = parts[3].capitalize()
            sentence = " ".join(parts)
        sentence = re.sub(r"\s+([,.;:!?])", r"\1", sentence)
        if sentence[-1] not in ".!?":
            sentence += "."
        return sentence

    def unavailable_rules(self, vocabulary: Sequence[str]) -> list[tuple[str, ...]]:
        """Return rule sources that cannot be emitted by any vocabulary-token sequence."""
        vocabulary_phrases = {
            tuple(unicodedata.normalize("NFC", token.replace("_", " ")).upper().split())
            for token in vocabulary
            if token and not token.startswith("<")
        }

        @lru_cache(maxsize=None)
        def can_emit(source: tuple[str, ...], offset: int) -> bool:
            if offset == len(source):
                return True
            return any(
                source[offset:offset + len(phrase)] == phrase
                and can_emit(source, offset + len(phrase))
                for phrase in vocabulary_phrases
                if phrase
            )

        return [
            source for source in self.reordering_rules
            if not can_emit(source, 0)
        ]

    @staticmethod
    def _normalize_rule_phrase(phrase: Sequence[str]) -> tuple[str, ...]:
        return tuple(unicodedata.normalize("NFC", word).upper().strip() for word in phrase if word.strip())

    def _reorder(self, words: list[str]) -> list[str]:
        reordered: list[str] = []
        index = 0
        while index < len(words):
            for source, target in self._ordered_rules:
                if tuple(words[index:index + len(source)]) == source:
                    reordered.extend(target)
                    index += len(source)
                    break
            else:
                reordered.append(words[index])
                index += 1
        return reordered
