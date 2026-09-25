"""Bounded global-workspace broadcast for TARA Baby.

Grounding: global-workspace theories propose that selected information can become
widely available to otherwise specialized systems. TARA uses that as a functional
architecture for cross-module broadcasting, without making a claim about machine
consciousness or subjective experience.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkspaceItem:
    content: str
    source: str
    priority: float

    def __post_init__(self) -> None:
        if not self.content.strip() or not self.source.strip():
            raise ValueError("content and source must be non-empty")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")


class GlobalWorkspace:
    """Keep a small set of highest-priority items and broadcast snapshots."""

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._items: list[WorkspaceItem] = []

    @property
    def items(self) -> tuple[WorkspaceItem, ...]:
        return tuple(self._items)

    def broadcast(self, item: WorkspaceItem) -> tuple[WorkspaceItem, ...]:
        self._items.append(item)
        self._items.sort(key=lambda x: (x.priority, x.source, x.content), reverse=True)
        del self._items[self.capacity :]
        return self.items

    def clear(self) -> None:
        self._items.clear()

    def snapshot(self) -> tuple[WorkspaceItem, ...]:
        return self.items
