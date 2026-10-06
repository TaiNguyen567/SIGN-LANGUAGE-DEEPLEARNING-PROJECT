"""Local SQLite history for recognized sessions; camera frames are never stored."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class HistoryStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS translations ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
                "raw_sign_sequence TEXT NOT NULL, translated_sentence TEXT NOT NULL, confidence REAL NOT NULL)"
            )

    def add(self, raw_sign_sequence: str, translated_sentence: str, confidence: float) -> None:
        if not raw_sign_sequence.strip():
            return
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO translations(timestamp, raw_sign_sequence, translated_sentence, confidence) VALUES (?, ?, ?, ?)",
                (timestamp, raw_sign_sequence, translated_sentence, float(confidence)),
            )

    def list_recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT timestamp, raw_sign_sequence, translated_sentence, confidence "
                "FROM translations ORDER BY id DESC LIMIT ?",
                (max(1, int(limit)),),
            ).fetchall()
        return [dict(zip(("timestamp", "raw_sign_sequence", "translated_sentence", "confidence"), row)) for row in rows]

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM translations")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)
