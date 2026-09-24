"""Quantitative scientific-reasoning primitives for TARA Baby.

The module favors explicit evidence, uncertainty and causal structure over
free-form assertions. It is designed for mathematical/scientific tasks and is
independent of the language model so exact procedures can be tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Measurement:
    value: float
    uncertainty: float
    unit: str = ""

    def __post_init__(self) -> None:
        if self.uncertainty < 0:
            raise ValueError("uncertainty must be >= 0")


@dataclass(frozen=True)
class Hypothesis:
    name: str
    prior: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("hypothesis name must be non-empty")
        if not 0.0 < self.prior < 1.0:
            raise ValueError("prior must be in (0, 1)")


@dataclass(frozen=True)
class Evidence:
    likelihood_if_true: float
    likelihood_if_false: float
    label: str = ""

    def __post_init__(self) -> None:
        if not 0.0 < self.likelihood_if_true <= 1.0:
            raise ValueError("likelihood_if_true must be in (0, 1]")
        if not 0.0 < self.likelihood_if_false <= 1.0:
            raise ValueError("likelihood_if_false must be in (0, 1]")


class BayesianReasoner:
    """Update hypothesis probability using sequential evidence."""

    def posterior(self, hypothesis: Hypothesis, evidence: list[Evidence]) -> float:
        odds = hypothesis.prior / (1.0 - hypothesis.prior)
        for item in evidence:
            odds *= item.likelihood_if_true / item.likelihood_if_false
        return odds / (1.0 + odds)

    def log_odds_update(self, hypothesis: Hypothesis, evidence: list[Evidence]) -> float:
        log_odds = math.log(hypothesis.prior / (1.0 - hypothesis.prior))
        for item in evidence:
            log_odds += math.log(item.likelihood_if_true / item.likelihood_if_false)
        return log_odds


@dataclass(frozen=True)
class CausalEdge:
    source: str
    target: str
    effect: float


class CausalGraph:
    """Small directed weighted graph for intervention-style linear effects."""

    def __init__(self) -> None:
        self._edges: dict[str, list[CausalEdge]] = {}

    def add_edge(self, source: str, target: str, effect: float) -> None:
        if not source or not target:
            raise ValueError("source and target must be non-empty")
        if source == target:
            raise ValueError("self-causal edge is not allowed")
        self._edges.setdefault(source, []).append(CausalEdge(source, target, float(effect)))

    def predict_intervention(
        self,
        intervention: dict[str, float],
        steps: int = 4,
    ) -> dict[str, float]:
        """Propagate a simple linear intervention through the causal graph."""
        if steps < 1:
            raise ValueError("steps must be >= 1")
        values = dict(intervention)
        frontier = dict(intervention)
        for _ in range(steps):
            next_frontier: dict[str, float] = {}
            for source, delta in frontier.items():
                for edge in self._edges.get(source, []):
                    target_delta = delta * edge.effect
                    next_frontier[edge.target] = next_frontier.get(edge.target, 0.0) + target_delta
            for target, delta in next_frontier.items():
                values[target] = values.get(target, 0.0) + delta
            frontier = next_frontier
            if not frontier:
                break
        return values


def weighted_measurement_mean(measurements: list[Measurement]) -> Measurement:
    """Combine independent measurements using inverse-variance weighting."""
    if not measurements:
        raise ValueError("at least one measurement is required")
    if any(m.uncertainty <= 0 for m in measurements):
        raise ValueError("all uncertainties must be > 0 for weighted mean")
    units = {m.unit for m in measurements if m.unit}
    if len(units) > 1:
        raise ValueError("measurement units must match")
    weights = [1.0 / (m.uncertainty * m.uncertainty) for m in measurements]
    total = sum(weights)
    mean = sum(m.value * w for m, w in zip(measurements, weights)) / total
    uncertainty = math.sqrt(1.0 / total)
    unit = next(iter(units)) if units else ""
    return Measurement(mean, uncertainty, unit)
