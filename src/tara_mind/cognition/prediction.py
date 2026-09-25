"""Prediction-error mechanisms for TARA Baby.

Grounding: predictive-processing research motivates maintaining hypotheses about
incoming observations and updating them when observations disagree. This module
keeps that idea explicit and testable without claiming that the implementation
is a literal brain model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Prediction:
    key: str
    predicted: float
    precision: float = 1.0

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("key must be non-empty")
        if self.precision <= 0:
            raise ValueError("precision must be > 0")


@dataclass(frozen=True)
class PredictionError:
    key: str
    observed: float
    predicted: float
    error: float
    weighted_error: float
    surprise: float


class PredictionEngine:
    """Compute signed/weighted prediction errors and surprise."""

    def compare(self, prediction: Prediction, observed: float) -> PredictionError:
        observed = float(observed)
        error = observed - prediction.predicted
        weighted = prediction.precision * error
        surprise = math.log1p(abs(weighted))
        return PredictionError(
            key=prediction.key,
            observed=observed,
            predicted=prediction.predicted,
            error=error,
            weighted_error=weighted,
            surprise=surprise,
        )

    @staticmethod
    def update(old_prediction: Prediction, observed: float, learning_rate: float = 0.1) -> Prediction:
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        new_value = old_prediction.predicted + learning_rate * (float(observed) - old_prediction.predicted)
        return Prediction(key=old_prediction.key, predicted=new_value, precision=old_prediction.precision)
