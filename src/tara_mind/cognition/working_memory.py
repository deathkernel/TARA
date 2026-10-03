"""Bounded working-memory mechanism for TARA Baby.

The design is inspired by cognitive models in which working memory maintains
and manipulates currently useful information under limited capacity.
This is an engineering hypothesis, not a literal reconstruction of the brain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class WorkingMemorySlot:
    """One active representation held in working memory."""

    content: Any
    source: str = "unknown"
    priority: float = 0.5
    access_count: int = 0
    age: int = 0

    def __post_init__(self) -> None:
        self.priority = float(self.priority)
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")
        if self.access_count < 0 or self.age < 0:
            raise ValueError("access_count and age must be >= 0")


class BoundedWorkingMemory:
    """Maintain a bounded set of active representations.

    Retention combines explicit priority, recency, and reuse. When capacity is
    exceeded, the lowest-retention representation is replaced.
    """

    def __init__(
        self,
        capacity: int = 7,
        recency_weight: float = 0.30,
        reuse_weight: float = 0.15,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if recency_weight < 0 or reuse_weight < 0:
            raise ValueError("weights must be non-negative")
        if recency_weight + reuse_weight > 1:
            raise ValueError("recency_weight + reuse_weight must be <= 1")
        self.capacity = int(capacity)
        self.recency_weight = float(recency_weight)
        self.reuse_weight = float(reuse_weight)
        self._slots: list[WorkingMemorySlot] = []

    @property
    def slots(self) -> tuple[WorkingMemorySlot, ...]:
        return tuple(self._slots)

    def _retention(self, slot: WorkingMemorySlot) -> float:
        recency = 1.0 / (1.0 + slot.age)
        reuse = min(1.0, slot.access_count / 5.0)
        priority_weight = 1.0 - self.recency_weight - self.reuse_weight
        return (
            priority_weight * slot.priority
            + self.recency_weight * recency
            + self.reuse_weight * reuse
        )

    def tick(self) -> None:
        """Advance internal time and age all active representations."""
        for slot in self._slots:
            slot.age += 1

    def attend(self, index: int) -> Any:
        """Refresh one item and increment its reuse count."""
        slot = self._slots[index]
        slot.age = 0
        slot.access_count += 1
        return slot.content

    def add(
        self,
        content: Any,
        source: str = "unknown",
        priority: float = 0.5,
    ) -> int:
        """Add a representation and return its slot index."""
        self.tick()
        slot = WorkingMemorySlot(
            content=content,
            source=source,
            priority=priority,
            age=0,
        )
        if len(self._slots) < self.capacity:
            self._slots.append(slot)
            return len(self._slots) - 1

        replacement = min(
            range(len(self._slots)),
            key=lambda i: self._retention(self._slots[i]),
        )
        self._slots[replacement] = slot
        return replacement

    def retain(self, limit: int | None = None) -> list[WorkingMemorySlot]:
        """Return active items ordered from highest to lowest retention."""
        items = sorted(self._slots, key=self._retention, reverse=True)
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be >= 0")
            items = items[:limit]
        return items

    def remove(self, index: int) -> WorkingMemorySlot:
        return self._slots.pop(index)

    def clear(self) -> None:
        self._slots.clear()

    def retention_scores(self) -> list[float]:
        return [self._retention(slot) for slot in self._slots]
