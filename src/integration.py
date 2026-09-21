"""Small orchestration layer connecting TARA's core cognitive modules.

This layer deliberately performs no PC automation. It combines perception,
memory, reasoning and language-model interfaces into an inspectable state
container so later agent/tool layers have a stable boundary.
"""

from dataclasses import dataclass, field

from .memory import WorkingMemory
from .reasoning import Goal, Plan, make_plan


@dataclass
class CognitiveState:
    """Current bounded state available to the reasoning system."""

    goal: Goal | None = None
    plan: Plan | None = None
    observations: WorkingMemory = field(default_factory=lambda: WorkingMemory(8))
    results: list[object] = field(default_factory=list)

    def observe(self, observation):
        self.observations.add(observation)

    def record_result(self, result):
        self.results.append(result)


class TARAEngine:
    """Coordinate explicit cognitive state without executing external actions."""

    def __init__(self, working_memory_capacity=8):
        self.state = CognitiveState(
            observations=WorkingMemory(working_memory_capacity)
        )

    def set_goal(self, description, success_conditions=()):
        self.state.goal = Goal(description, tuple(success_conditions))
        self.state.plan = None
        self.state.results.clear()
        return self.state.goal

    def plan(self, subtasks):
        if self.state.goal is None:
            raise ValueError("a goal must be set before planning")
        self.state.plan = make_plan(self.state.goal, subtasks)
        return self.state.plan

    def observe(self, observation):
        self.state.observe(observation)

    def record_result(self, result):
        self.state.record_result(result)

    def snapshot(self):
        """Return a serializable, concise snapshot of cognitive state."""
        return {
            "goal": None if self.state.goal is None else self.state.goal.description,
            "plan": None if self.state.plan is None else [
                task.description for task in self.state.plan.steps
            ],
            "current_step": None if self.state.plan is None else self.state.plan.current_index,
            "observations": self.state.observations.recent(),
            "results": list(self.state.results),
        }
