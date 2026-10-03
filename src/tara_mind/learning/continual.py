"""Continual-learning coordinator for TARA Baby.

Grounding: replay and consolidation can support learning from new experience
while protecting previously useful knowledge. This module coordinates existing
memory mechanisms; it does not directly modify Transformer weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..memory.consolidation import ReplayBuffer, ReplayItem
from ..memory.episodic import Episode
from ..memory.system import ConsolidationReport, MemorySystem


@dataclass(frozen=True)
class LearningReport:
    replayed: int
    consolidation: ConsolidationReport


class ContinualLearner:
    """Route experiences through replay and semantic/schema consolidation."""

    def __init__(self, memory: MemorySystem | None = None, replay_capacity: int = 256) -> None:
        self.memory = memory or MemorySystem()
        self.replay = ReplayBuffer(replay_capacity)

    def learn_episode(self, episode: Episode) -> None:
        self.memory.remember(episode)
        self.replay.add(
            ReplayItem(
                episode_id=episode.episode_id,
                payload=" | ".join(episode.events),
                prediction_error=episode.prediction_error,
                importance=episode.importance,
                novelty=episode.novelty,
            )
        )

    def consolidate(self, replay_batch: int = 16, memory_batch: int = 32) -> LearningReport:
        if replay_batch < 0 or memory_batch < 0:
            raise ValueError("batch sizes must be >= 0")
        replayed = len(self.replay.sample(replay_batch))
        report = self.memory.consolidate(memory_batch)
        return LearningReport(replayed, report)
