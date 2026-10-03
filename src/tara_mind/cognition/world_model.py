"""Integrated world-state model for TARA Baby.

This module connects learned transition structure to prediction error. It is
inspired by predictive-processing and hippocampal predictive-map ideas, while
remaining a small inspectable computational hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .predictive_map import PredictiveMap


@dataclass(frozen=True)
class WorldObservation:
    state: str
    next_state: str


@dataclass(frozen=True)
class WorldUpdate:
    observation: WorldObservation
    predicted_state: str | None
    probability_of_observation: float
    surprise: float


@dataclass
class WorldModel:
    """Maintain a symbolic state-transition model and online surprise signal."""

    predictive_map: PredictiveMap = field(default_factory=PredictiveMap)
    current_state: str | None = None

    def observe(self, state: str, next_state: str) -> WorldUpdate:
        if not state or not next_state:
            raise ValueError("states must be non-empty")

        distribution = self.predictive_map.transition_distribution(state)
        predicted = max(distribution, key=distribution.get) if distribution else None
        # Before any evidence exists, use a neutral prior. After learning, use
        # the model probability and convert it to information-theoretic surprise.
        probability = distribution.get(next_state, 0.5) if distribution else 0.5
        probability = min(1.0, max(1e-12, probability))
        surprise = -math.log(probability)

        self.predictive_map.observe_transition(state, next_state)
        self.current_state = next_state
        return WorldUpdate(
            observation=WorldObservation(state, next_state),
            predicted_state=predicted,
            probability_of_observation=probability,
            surprise=surprise,
        )

    def predict(self, state: str | None = None) -> str | None:
        target = state if state is not None else self.current_state
        if target is None:
            return None
        return self.predictive_map.predict_next(target)
