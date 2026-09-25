"""Science-grounded selective-attention controller for TARA Baby.

This module is a computational hypothesis inspired by selective attention:
processing resources should be allocated preferentially to information that
is relevant to the current goal, salient/novel, urgent, uncertain, or needed
for correction. It is not a literal model of a brain attention circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class AttentionItem:
    """An observable candidate for attention allocation."""

    content: str
    goal_relevance: float = 0.0
    salience: float = 0.0
    novelty: float = 0.0
    urgency: float = 0.0
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string")
        for name in ("goal_relevance", "salience", "novelty", "urgency", "uncertainty"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class AttentionWeights:
    """Weights for a testable relevance-allocation hypothesis."""

    goal_relevance: float = 0.35
    salience: float = 0.20
    novelty: float = 0.15
    urgency: float = 0.15
    uncertainty: float = 0.15

    def __post_init__(self) -> None:
        values = (self.goal_relevance, self.salience, self.novelty, self.urgency, self.uncertainty)
        if any(value < 0.0 for value in values):
            raise ValueError("attention weights must be non-negative")
        if not math.isclose(sum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("attention weights must sum to 1")


@dataclass(frozen=True)
class AttendedItem:
    item: AttentionItem
    score: float


class SelectiveAttention:
    """Rank and allocate a bounded attention budget."""

    def __init__(self, weights: AttentionWeights | None = None, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.weights = weights or AttentionWeights()
        self.temperature = float(temperature)

    def score(self, item: AttentionItem) -> float:
        w = self.weights
        return (
            w.goal_relevance * item.goal_relevance
            + w.salience * item.salience
            + w.novelty * item.novelty
            + w.urgency * item.urgency
            + w.uncertainty * item.uncertainty
        )

    def allocate(self, items: Iterable[AttentionItem], budget: int) -> list[AttendedItem]:
        candidates = list(items)
        if budget < 0:
            raise ValueError("budget must be >= 0")
        if budget == 0 or not candidates:
            return []
        scaled = [self.score(item) / self.temperature for item in candidates]
        maximum = max(scaled)
        exp_scores = [math.exp(value - maximum) for value in scaled]
        total = sum(exp_scores)
        probabilities = [value / total for value in exp_scores]
        ranked = [AttendedItem(item=item, score=p) for item, p in zip(candidates, probabilities)]
        ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked[: min(budget, len(ranked))]
