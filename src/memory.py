"""Non-persistent compatibility interface for TARA's removed conversation memory.

This module intentionally stores no conversation data in memory or on disk.
The legacy class name remains as a no-op adapter for runtime components that
still accept a memory dependency; all recall operations return empty results.
"""


class LongTermMemory:
    """Disabled adapter: never retains, loads, or saves conversation content."""

    def __init__(self, path=None):
        # Accept the legacy argument for source compatibility, but never access it.
        self.path = None

    def remember(self, key, value, importance=1.0):
        return value

    def recall(self, key, default=None):
        return default

    def retrieve(self, query, limit=None, min_score=0.0):
        return []

    def forget(self, key):
        return None

    def forget_least_used(self, max_records):
        return []

    def keys(self):
        return []

    def metadata(self, key):
        return None

    def save(self, path=None):
        raise RuntimeError("persistent conversation memory has been removed")

    def load(self, path=None):
        raise RuntimeError("persistent conversation memory has been removed")


class WorkingMemory:
    """Small in-process buffer only; it is never persisted."""

    def __init__(self, capacity=8):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items = []

    def add(self, item):
        self._items.append(item)
        if len(self._items) > self.capacity:
            del self._items[0]

    def recent(self, limit=None):
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        return list(self._items if limit is None else self._items[-limit:] if limit else [])

    def clear(self):
        self._items.clear()

    def __len__(self):
        return len(self._items)
