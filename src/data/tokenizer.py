"""Unicode-aware word tokenizer for sign-sequence transcripts."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


class SignTokenizer:
    """Map Vietnamese words and punctuation to stable integer token IDs."""

    BLANK_TOKEN = "<blank>"
    UNKNOWN_TOKEN = "<unk>"
    TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

    def __init__(self, vocabulary: Sequence[str] | None = None) -> None:
        tokens = list(vocabulary or (self.BLANK_TOKEN, self.UNKNOWN_TOKEN))
        if len(tokens) < 2 or tokens[0] != self.BLANK_TOKEN or tokens[1] != self.UNKNOWN_TOKEN:
            raise ValueError("Vocabulary must start with <blank>, <unk>")
        if len(tokens) != len(set(tokens)):
            raise ValueError("Vocabulary contains duplicate tokens")
        self.tokens = tokens
        self.token_to_id = {token: index for index, token in enumerate(tokens)}

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def unknown_id(self) -> int:
        return 1

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFC", text).upper().strip()
        return cls.TOKEN_PATTERN.findall(normalized)

    @classmethod
    def build(cls, texts: Iterable[str], min_frequency: int = 1) -> "SignTokenizer":
        if min_frequency < 1:
            raise ValueError("min_frequency must be at least 1")
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(cls.tokenize(text))
        words = sorted(
            (token for token, count in counts.items() if count >= min_frequency),
            key=lambda token: (-counts[token], token),
        )
        return cls((cls.BLANK_TOKEN, cls.UNKNOWN_TOKEN, *words))

    def encode(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", str(text)).strip()
        if normalized in self.token_to_id:
            return [self.token_to_id[normalized]]
        upper = normalized.upper()
        if upper in self.token_to_id:
            return [self.token_to_id[upper]]
        return [self.token_to_id.get(token, self.unknown_id) for token in self.tokenize(text)]

    def decode(self, token_ids: Iterable[int], *, skip_blank: bool = True) -> str:
        words: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            if index < 0 or index >= len(self.tokens):
                token = self.UNKNOWN_TOKEN
            else:
                token = self.tokens[index]
            if skip_blank and token == self.BLANK_TOKEN:
                continue
            words.append(token.replace("_", " "))
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(words))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"tokens": self.tokens}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "SignTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not isinstance(payload.get("tokens"), list):
            raise ValueError("Invalid vocabulary file: expected a JSON object with a tokens list")
        return cls(payload["tokens"])
