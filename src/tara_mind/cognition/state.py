"""Core cognitive state representations for TARA Baby.

These structures are deliberately small and inspectable. They represent
computational state; they do not claim to reproduce human subjective
experience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class AffectState:
    """Behavioral appraisal signals inferred from the interaction context."""

    valence: float = 0.0
    arousal: float = 0.0
    urgency: float = 0.0
    frustration: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        for name in ("valence", "arousal", "urgency", "frustration", "confidence"):
            value = float(getattr(self, name))
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")


@dataclass
class WorkingMemoryItem:
    """One currently relevant item held in working memory."""

    content: Any
    source: str = "unknown"
    importance: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.importance = float(self.importance)
        if not 0.0 <= self.importance <= 1.0:
            raise ValueError("importance must be in [0, 1]")


@dataclass
class CognitiveState:
    """Explicit state passed through TARA Baby's cognitive loop."""

    goal: str | None = None
    context: list[WorkingMemoryItem] = field(default_factory=list)
    affect: AffectState = field(default_factory=AffectState)
    beliefs: dict[str, Any] = field(default_factory=dict)
    plan: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)

    def remember_working(self, content: Any, source: str = "unknown", importance: float = 0.5) -> None:
        self.context.append(
            WorkingMemoryItem(content=content, source=source, importance=importance)
        )

    def observe(self, message: str) -> None:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("observation must be a non-empty string")
        self.observations.append(message.strip())

    def clear_plan(self) -> None:
        self.plan.clear()
