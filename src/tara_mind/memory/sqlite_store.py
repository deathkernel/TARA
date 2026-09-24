"""Local SQLite backend for TARA Baby durable memory."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .store import MemoryRecord


class SQLiteMemoryStore:
    """Implement the memory-store contract using only Python's stdlib."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS memories (key TEXT PRIMARY KEY, value TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL)"
        )
        self._connection.execute("CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind)")
        self._connection.commit()

    def put(self, record: MemoryRecord) -> None:
        self._connection.execute(
            "INSERT INTO memories(key, value, kind, source) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, kind=excluded.kind, source=excluded.source",
            (record.key, record.value, record.kind, record.source),
        )
        self._connection.commit()

    def get(self, key: str) -> MemoryRecord | None:
        row = self._connection.execute(
            "SELECT key, value, kind, source FROM memories WHERE key = ?", (key,)
        ).fetchone()
        return MemoryRecord(*row) if row else None

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        pattern = f"%{query.strip()}%"
        rows = self._connection.execute(
            "SELECT key, value, kind, source FROM memories "
            "WHERE key LIKE ? OR value LIKE ? ORDER BY key LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [MemoryRecord(*row) for row in rows]

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteMemoryStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
