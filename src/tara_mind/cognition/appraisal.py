"""Affect/appraisal controller for TARA Baby.

Scientific grounding: appraisal research treats emotion-related responses as
sensitive to evaluations of relevance, goal congruence, control and uncertainty.
This module implements observable control signals for behavior; it does not
claim subjective feeling or biological equivalence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AppraisalInput:
    goal_relevance: float = 0.0
    goal_congruence: float = 0.0
    controllability: float = 0.0
    certainty: float = 0.0
    novelty: float = 0.0
    social_relevance: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "goal_relevance",
            "goal_congruence",
            "controllability",
            "certainty",
            "novelty",
            "social_relevance",
        ):
            value = float(getattr(self, name))
            if -1.0 > value or value > 1.0:
                raise ValueError(f"{name} must be in [-1, 1]")


@dataclass(frozen=True)
class AppraisalState:
    valence: float
    arousal: float
    urgency: float
    approach: float
    inhibition: float
    confidence: float


class AppraisalEngine:
    """Transform contextual appraisal variables into behavioral control signals."""

    def evaluate(self, x: AppraisalInput) -> AppraisalState:
        valence = 0.45 * x.goal_congruence + 0.20 * x.controllability + 0.20 * x.social_relevance + 0.15 * x.certainty
        arousal = min(1.0, 0.45 * abs(x.novelty) + 0.35 * abs(x.goal_relevance) + 0.20 * (1.0 - x.certainty))
        urgency = min(1.0, 0.55 * abs(x.goal_relevance) + 0.25 * abs(x.goal_congruence) + 0.20 * arousal)
        approach = max(-1.0, min(1.0, valence * (0.5 + 0.5 * x.controllability)))
        inhibition = max(0.0, min(1.0, 0.5 * (1.0 - x.controllability) + 0.35 * (1.0 - x.certainty) + 0.15 * abs(min(0.0, valence))))
        confidence = max(-1.0, min(1.0, 0.65 * x.certainty + 0.35 * x.controllability))
        return AppraisalState(valence, arousal, urgency, approach, inhibition, confidence)

    @staticmethod
    def regulate(state: AppraisalState, step: float = 0.15) -> AppraisalState:
        """Apply a bounded damping step to reduce extreme control signals."""
        if not 0.0 < step <= 1.0:
            raise ValueError("step must be in (0, 1]")
        return AppraisalState(
            valence=state.valence * (1.0 - step),
            arousal=state.arousal * (1.0 - step),
            urgency=state.urgency * (1.0 - step),
            approach=state.approach * (1.0 - step),
            inhibition=state.inhibition + step * (1.0 - state.inhibition),
            confidence=state.confidence,
        )
