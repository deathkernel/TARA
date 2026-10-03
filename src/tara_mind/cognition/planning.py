"""Model-based planning for TARA Baby.

Grounding: human planning can use internal models of future states, and
successor-style representations provide a computational abstraction of future
occupancy. TARA uses explicit transition probabilities to perform bounded mental
simulation before choosing an action. This is not a literal brain simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .predictive_map import PredictiveMap


@dataclass(frozen=True)
class Plan:
    states: tuple[str, ...]
    log_probability: float
    score: float

    @property
    def probability(self) -> float:
        return math.exp(self.log_probability)


class ModelBasedPlanner:
    """Bounded beam-search planner over the learned transition model."""

    def __init__(self, predictive_map: PredictiveMap, beam_width: int = 8) -> None:
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        self.model = predictive_map
        self.beam_width = beam_width

    def plan(
        self,
        start: str,
        goal: str,
        max_depth: int = 8,
        state_values: dict[str, float] | None = None,
    ) -> Plan | None:
        if not start or not goal:
            raise ValueError("start and goal must be non-empty")
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        values = state_values or {goal: 1.0}
        frontier: list[tuple[tuple[str, ...], float, float]] = [((start,), 0.0, 0.0)]
        best_goal: Plan | None = Plan((start,), 0.0, values.get(start, 0.0)) if start == goal else None

        for _ in range(max_depth):
            candidates: list[tuple[tuple[str, ...], float, float]] = []
            for path, log_probability, score in frontier:
                current = path[-1]
                for nxt, probability in self.model.transition_distribution(current).items():
                    if nxt in path:
                        continue
                    new_log = log_probability + math.log(max(probability, 1e-12))
                    new_score = score + values.get(nxt, 0.0)
                    new_path = path + (nxt,)
                    if nxt == goal:
                        candidate = Plan(new_path, new_log, new_score)
                        if best_goal is None or (candidate.score, candidate.log_probability) > (best_goal.score, best_goal.log_probability):
                            best_goal = candidate
                    candidates.append((new_path, new_log, new_score))
            if not candidates:
                break
            candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
            frontier = candidates[: self.beam_width]
        return best_goal
