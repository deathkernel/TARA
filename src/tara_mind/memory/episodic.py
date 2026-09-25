"""Structured episodic memory for TARA Baby.

Scientific grounding: episodic memory binds an event with contextual and temporal
details. Hippocampal research motivates an explicit event store while avoiding a
literal claim that this data structure reproduces hippocampal circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable


@dataclass(frozen=True)
class Episode:
    episode_id: str
    timestamp: str
    context: tuple[str, ...] = ()
    events: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    facts: tuple[str, ...] = ()
    prediction_error: float = 0.0
    importance: float = 0.5
    novelty: float = 0.5

    @classmethod
    def create(
        cls,
        episode_id: str,
        events: Iterable[str],
        context: Iterable[str] = (),
        entities: Iterable[str] = (),
        facts: Iterable[str] = (),
        prediction_error: float = 0.0,
        importance: float = 0.5,
        novelty: float = 0.5,
    ) -> "Episode":
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        values = {"prediction_error": prediction_error, "importance": importance, "novelty": novelty}
        for name, value in values.items():
            if float(value) < 0:
                raise ValueError(f"{name} must be >= 0")
        event_tuple = tuple(str(x).strip() for x in events if str(x).strip())
        if not event_tuple:
            raise ValueError("episode must contain at least one event")
        return cls(
            episode_id=episode_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            context=tuple(str(x).strip() for x in context if str(x).strip()),
            events=event_tuple,
            entities=tuple(str(x).strip() for x in entities if str(x).strip()),
            facts=tuple(str(x).strip() for x in facts if str(x).strip()),
            prediction_error=float(prediction_error),
            importance=float(importance),
            novelty=float(novelty),
        )

    def replay_priority(self) -> float:
        return (
            0.50 * self.prediction_error
            + 0.30 * self.importance
            + 0.20 * self.novelty
        )


class EpisodicMemory:
    """Bounded event memory that preserves episodes with contextual detail."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self._episodes: dict[str, Episode] = {}

    def store(self, episode: Episode) -> None:
        self._episodes[episode.episode_id] = episode
        if len(self._episodes) > self.capacity:
            weakest = min(
                self._episodes.values(),
                key=lambda e: (e.replay_priority(), e.timestamp, e.episode_id),
            )
            self._episodes.pop(weakest.episode_id)

    def get(self, episode_id: str) -> Episode | None:
        return self._episodes.get(episode_id)

    def retrieve(self, query_terms: Iterable[str], limit: int = 5) -> list[Episode]:
        terms = {str(t).lower().strip() for t in query_terms if str(t).strip()}
        if limit < 0:
            raise ValueError("limit must be >= 0")

        def score(episode: Episode) -> tuple[int, float, str]:
            searchable = {x.lower() for x in episode.context + episode.events + episode.entities + episode.facts}
            overlap = len(terms & searchable)
            return overlap, episode.replay_priority(), episode.timestamp

        ranked = sorted(self._episodes.values(), key=score, reverse=True)
        return ranked[:limit]

    def __len__(self) -> int:
        return len(self._episodes)
