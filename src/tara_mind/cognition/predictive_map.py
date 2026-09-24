"""Successor-style predictive map for TARA Baby.

Grounding: hippocampal research links predictive representations to spatial,
contextual and sequential structure. Successor representations provide a useful
computational abstraction: represent expected future-state occupancy rather
than only the current state.

This is a discrete research prototype, not a literal hippocampus simulation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Transition:
    state: str
    next_state: str
    probability: float


class PredictiveMap:
    """Learn transition statistics and estimate discounted future occupancy."""

    def __init__(self, gamma: float = 0.9, smoothing: float = 1.0) -> None:
        if not 0.0 < gamma < 1.0:
            raise ValueError("gamma must be in (0, 1)")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0")
        self.gamma = float(gamma)
        self.smoothing = float(smoothing)
        self._counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def observe_transition(self, state: str, next_state: str) -> None:
        if not state or not next_state:
            raise ValueError("states must be non-empty")
        self._counts[state][next_state] += 1.0

    def transition_distribution(self, state: str) -> dict[str, float]:
        row = self._counts.get(state, {})
        if not row:
            return {}
        states = sorted(row)
        total = sum(row.values()) + self.smoothing * len(states)
        return {s: (row[s] + self.smoothing) / total for s in states}

    def predict_next(self, state: str) -> str | None:
        distribution = self.transition_distribution(state)
        return max(distribution, key=distribution.get) if distribution else None

    def successor_features(self, start_state: str, states: list[str] | None = None, iterations: int = 200) -> dict[str, float]:
        """Approximate M = I + gamma*T*M by fixed-point iteration."""
        if iterations < 1:
            raise ValueError("iterations must be >= 1")

        known = set(states or [])
        known.add(start_state)
        for source, row in self._counts.items():
            known.add(source)
            known.update(row)
        ordered = sorted(known)
        index = {state: i for i, state in enumerate(ordered)}
        n = len(ordered)

        transition = [[0.0] * n for _ in range(n)]
        for source in ordered:
            for target, probability in self.transition_distribution(source).items():
                transition[index[source]][index[target]] = probability

        current = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for _ in range(iterations):
            new = [[0.0] * n for _ in range(n)]
            max_delta = 0.0
            for i in range(n):
                for j in range(n):
                    value = (1.0 if i == j else 0.0) + self.gamma * sum(
                        transition[i][k] * current[k][j] for k in range(n)
                    )
                    new[i][j] = value
                    max_delta = max(max_delta, abs(value - current[i][j]))
            current = new
            if max_delta < 1e-8:
                break

        row = current[index[start_state]]
        return {state: row[index[state]] for state in ordered}
