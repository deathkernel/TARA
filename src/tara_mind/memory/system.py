"""Unified memory system for TARA Baby.

The system reflects a computational version of a hippocampal/semantic-memory
interaction: preserve detailed episodes, extract generalized knowledge from
repeated experience, and use surprise/importance to prioritize consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass

from .episodic import Episode, EpisodicMemory
from .schema import SchemaLearner
from .semantic import SemanticMemory


@dataclass(frozen=True)
class ConsolidationReport:
    episodes_considered: int
    facts_learned: int
    schemas_learned: int


class MemorySystem:
    """Coordinate episodic storage, semantic consolidation and schemas."""

    def __init__(self, episodic_capacity: int = 4096, schema_min_support: int = 2) -> None:
        self.episodic = EpisodicMemory(episodic_capacity)
        self.semantic = SemanticMemory()
        self.schemas = SchemaLearner(schema_min_support)

    def remember(self, episode: Episode) -> None:
        self.episodic.store(episode)

    def consolidate(self, limit: int = 32) -> ConsolidationReport:
        episodes = sorted(
            self.episodic._episodes.values(),
            key=lambda e: (-e.replay_priority(), e.timestamp, e.episode_id),
        )[: max(0, limit)]
        facts_before = len(self.semantic)
        schema_before = len(self.schemas.schemas())

        for episode in episodes:
            confidence = min(1.0, max(0.1, 0.5 + 0.25 * episode.importance + 0.25 * episode.novelty))
            for fact in episode.facts:
                self.semantic.learn(fact, "supported by remembered experience", confidence, episode.episode_id)
            if episode.context:
                self.schemas.observe(episode.context[0], episode.events)

        return ConsolidationReport(
            episodes_considered=len(episodes),
            facts_learned=len(self.semantic) - facts_before,
            schemas_learned=len(self.schemas.schemas()) - schema_before,
        )

    def retrieve(self, query_terms: list[str], query: str) -> dict[str, object]:
        return {
            "episodes": self.episodic.retrieve(query_terms),
            "facts": self.semantic.search(query),
        }
