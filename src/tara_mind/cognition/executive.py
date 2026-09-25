"""Executive-control mechanism for TARA Baby.

Scientific grounding: contemporary neuroscience links goal-directed behavior
to interactions among prefrontal and striatal systems, while inhibitory control
supports stopping or withholding actions. TARA implements an explicit symbolic
controller that can prioritize goals, compare candidate actions, and block risky
or conflict-heavy actions before tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Goal:
    goal_id: str
    description: str
    priority: float = 0.5
    deadline: float | None = None

    def __post_init__(self) -> None:
        if not self.goal_id or not self.description:
            raise ValueError("goal_id and description must be non-empty")
        if not 0.0 <= self.priority <= 1.0:
            raise ValueError("priority must be in [0, 1]")


@dataclass(frozen=True)
class ActionCandidate:
    action_id: str
    goal_id: str
    expected_value: float = 0.0
    expected_cost: float = 0.0
    risk: float = 0.0
    confidence: float = 0.0
    requires_confirmation: bool = False

    def __post_init__(self) -> None:
        if not self.action_id or not self.goal_id:
            raise ValueError("action_id and goal_id must be non-empty")
        for name in ("risk", "confidence"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class Decision:
    action_id: str | None
    score: float
    blocked: bool
    reason: str


@dataclass
class ExecutiveController:
    """Goal stack plus an interpretable action-selection and inhibition gate."""

    risk_tolerance: float = 0.35
    goals: list[Goal] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not 0.0 <= self.risk_tolerance <= 1.0:
            raise ValueError("risk_tolerance must be in [0, 1]")

    def add_goal(self, goal: Goal) -> None:
        self.goals.append(goal)

    def active_goal(self) -> Goal | None:
        return max(self.goals, key=lambda goal: (goal.priority, goal.goal_id), default=None)

    def choose(self, actions: list[ActionCandidate]) -> Decision:
        if not actions:
            return Decision(None, 0.0, True, "no candidate actions")
        goal = self.active_goal()
        candidates = [a for a in actions if goal is None or a.goal_id == goal.goal_id]
        if not candidates:
            return Decision(None, 0.0, True, "no action matches active goal")

        def score(action: ActionCandidate) -> float:
            return action.expected_value + action.confidence - action.expected_cost - action.risk

        ranked = sorted(candidates, key=lambda action: (score(action), action.action_id), reverse=True)
        best = ranked[0]
        if best.risk > self.risk_tolerance:
            return Decision(best.action_id, score(best), True, "risk exceeds executive tolerance")
        if best.requires_confirmation:
            return Decision(best.action_id, score(best), True, "confirmation required")
        if score(best) < 0.0:
            return Decision(best.action_id, score(best), True, "expected utility is negative")
        return Decision(best.action_id, score(best), False, "selected")

    def inhibit(self, decision: Decision, conflict: bool) -> Decision:
        if conflict and not decision.blocked:
            return Decision(decision.action_id, decision.score, True, "inhibited by conflict monitoring")
        return decision
