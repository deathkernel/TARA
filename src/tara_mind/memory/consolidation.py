"""Memory replay and consolidation priorities for TARA Baby.

Grounding: hippocampal replay during awake and sleep-related states is associated
with memory consolidation and transformation. TARA cannot reproduce biological
sleep physiology, so this module implements a controllable replay abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplayItem:
    episode_id: str
    payload: str
    prediction_error: float = 0.0
    importance: float = 0.5
    novelty: float = 0.5
    replay_count: int = 0

    def priority(self) -> float:
        return (
            0.50 * max(0.0, self.prediction_error)
            + 0.30 * max(0.0, self.importance)
            + 0.20 * max(0.0, self.novelty)
        ) / (1.0 + 0.15 * self.replay_count)


class ReplayBuffer:
    """Priority-ordered episodic replay buffer."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = int(capacity)
        self._items: dict[str, ReplayItem] = {}

    def add(self, item: ReplayItem) -> None:
        self._items[item.episode_id] = item
        if len(self._items) > self.capacity:
            weakest = min(self._items.values(), key=lambda x: (x.priority(), x.episode_id))
            self._items.pop(weakest.episode_id)

    def sample(self, k: int = 1) -> list[ReplayItem]:
        if k < 0:
            raise ValueError("k must be >= 0")
        ranked = sorted(self._items.values(), key=lambda x: (-x.priority(), x.episode_id))
        chosen = ranked[: min(k, len(ranked))]
        for item in chosen:
            self._items[item.episode_id] = ReplayItem(
                episode_id=item.episode_id,
                payload=item.payload,
                prediction_error=item.prediction_error,
                importance=item.importance,
                novelty=item.novelty,
                replay_count=item.replay_count + 1,
            )
        return chosen

    def __len__(self) -> int:
        return len(self._items)
