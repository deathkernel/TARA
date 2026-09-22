"""Basic goal progress tracking for TARA."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Progress:
    goal: str
    completed: int
    total: int
    ratio: float
    status: str


class GoalProgress:
    """Track simple verified milestones for a goal."""

    def __init__(self):
        self._goals = {}

    def start(self, goal, total=1):
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        if total <= 0:
            raise ValueError("total must be positive")
        self._goals[goal] = {"completed": 0, "total": int(total)}
        return self.snapshot(goal)

    def mark_complete(self, goal, count=1):
        if goal not in self._goals:
            raise KeyError(goal)
        if count < 0:
            raise ValueError("count must be non-negative")
        state = self._goals[goal]
        state["completed"] = min(state["total"], state["completed"] + int(count))
        return self.snapshot(goal)

    def snapshot(self, goal):
        if goal not in self._goals:
            raise KeyError(goal)
        state = self._goals[goal]
        ratio = state["completed"] / state["total"]
        status = "complete" if ratio >= 1.0 else "in-progress"
        return Progress(goal, state["completed"], state["total"], ratio, status)
