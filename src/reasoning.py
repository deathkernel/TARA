"""Small, deterministic reasoning primitives for TARA.

Research basis:
- Yao et al. (2023), Tree of Thoughts: deliberate reasoning can explore and
  evaluate alternative intermediate paths.
- Yao et al. (2023), ReAct: reasoning benefits from explicit task/subgoal
  tracking and feedback between steps.
- Hao et al. (2023), RAP: planning can be treated as search over candidate
  reasoning/action paths with state-aware evaluation.

TARA starts with an explicit symbolic layer rather than a learned planner.
This keeps goal representation, decomposition, planning, verification, and
re-planning inspectable and testable before connecting them to the neural core.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Goal:
    """A structured desired outcome."""

    description: str
    success_conditions: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("goal description must be a non-empty string")
        if any(not isinstance(condition, str) or not condition.strip()
               for condition in self.success_conditions):
            raise ValueError("success conditions must be non-empty strings")


@dataclass(frozen=True)
class Task:
    """A single reasoning/planning unit."""

    description: str
    parent: str | None = None

    def __post_init__(self):
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("task description must be a non-empty string")
        if self.parent is not None and not isinstance(self.parent, str):
            raise ValueError("task parent must be a string or None")


@dataclass
class Plan:
    """Ordered executable reasoning steps."""

    goal: Goal
    steps: list[Task] = field(default_factory=list)
    current_index: int = 0

    def __post_init__(self):
        if self.current_index < 0 or self.current_index > len(self.steps):
            raise ValueError("current_index must point inside the plan")

    @property
    def complete(self):
        return self.current_index >= len(self.steps)

    @property
    def current(self):
        return None if self.complete else self.steps[self.current_index]

    def advance(self):
        if not self.complete:
            self.current_index += 1
        return self.current


def decompose_goal(goal, subtasks):
    """Convert a goal into an ordered list of explicit subtasks."""
    if not isinstance(goal, Goal):
        raise TypeError("goal must be a Goal")
    if not subtasks:
        raise ValueError("subtasks must not be empty")
    tasks = [task if isinstance(task, Task) else Task(str(task), goal.description)
             for task in subtasks]
    return tasks


def make_plan(goal, subtasks):
    """Build a deterministic sequential plan from decomposed subtasks."""
    return Plan(goal=goal, steps=decompose_goal(goal, subtasks))


def verify_step(expected, observed):
    """Verify a step using an explicit expected/observed equality contract."""
    if expected is None:
        raise ValueError("expected result must not be None")
    return expected == observed


def verify_goal(goal, state):
    """Check explicit goal conditions against a state mapping.

    Conditions are represented by keys whose values must be truthy. This is
    deliberately simple: a future learned verifier can implement richer
    semantic checks behind the same interface.
    """
    if not isinstance(goal, Goal):
        raise TypeError("goal must be a Goal")
    if not isinstance(state, dict):
        raise TypeError("state must be a dictionary")
    return all(bool(state.get(condition, False)) for condition in goal.success_conditions)


def replan(plan, failed_index, replacement_tasks):
    """Replace the failed step and all later steps with a new deterministic path."""
    if not isinstance(plan, Plan):
        raise TypeError("plan must be a Plan")
    if not 0 <= failed_index < len(plan.steps):
        raise IndexError("failed_index out of range")
    if not replacement_tasks:
        raise ValueError("replacement_tasks must not be empty")

    replacements = [
        task if isinstance(task, Task) else Task(str(task), plan.goal.description)
        for task in replacement_tasks
    ]
    plan.steps = plan.steps[:failed_index] + replacements
    plan.current_index = failed_index
    return plan
