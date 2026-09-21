"""Small memory primitives for TARA.

Research basis:
- Graves et al. (2014), Neural Turing Machines: neural systems can be paired
  with external memory through an explicit read/write interface.
- Lewis et al. (2020), Retrieval-Augmented Generation: non-parametric external
  memory can complement parametric model knowledge.

TARA starts with two intentionally simple stores: bounded working memory for
recent context and persistent long-term memory for durable facts. Retrieval
ranking is deliberately deferred to a later milestone.
"""

import json
from collections import deque
from pathlib import Path


class WorkingMemory:
    """Bounded FIFO working memory for recent observations or thoughts."""

    def __init__(self, capacity=8):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items = deque(maxlen=self.capacity)

    def add(self, item):
        self._items.append(item)

    def recent(self, limit=None):
        """Return recent items in oldest-to-newest order."""
        items = list(self._items)
        if limit is None:
            return items
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return items[-limit:] if limit else []

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)


class LongTermMemory:
    """Simple persistent key/value memory stored as human-readable JSON."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._records = {}
        if self.path is not None and self.path.exists():
            self.load()

    def remember(self, key, value):
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")
        self._records[key] = value

    def recall(self, key, default=None):
        return self._records.get(key, default)

    def forget(self, key):
        return self._records.pop(key, None)

    def keys(self):
        return list(self._records.keys())

    def save(self, path=None):
        destination = Path(path) if path is not None else self.path
        if destination is None:
            raise ValueError("a path is required to save long-term memory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self._records, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.path = destination

    def load(self, path=None):
        source = Path(path) if path is not None else self.path
        if source is None:
            raise ValueError("a path is required to load long-term memory")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("long-term memory must contain a JSON object")
        self._records = payload
        self.path = source
