"""Minimal durable-memory interface for TARA Baby.

The implementation is intentionally storage-agnostic. A concrete SQLite/vector
store can implement this interface without changing the cognitive loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
    kind: str = "fact"
    source: str = "unknown"


class MemoryStore(Protocol):
    def put(self, record: MemoryRecord) -> None: ...
    def get(self, key: str) -> MemoryRecord | None: ...
    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]: ...
