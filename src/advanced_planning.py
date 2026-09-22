"""Advanced planning primitives for TARA.

The planner is deliberately explicit and testable. It supports goal
 decomposition, dependency-aware multi-step plans, validation, alternatives,
 simple resource/cost accounting and plan revision from observed failures.
It is a planning substrate for a future learned planner, not a claim of
perfect autonomous planning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, Mapping


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    step_id: str
    description: str
    dependencies: tuple[str, ...] = ()
    cost: float = 1.0
    priority: int = 0
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    result: object | None = None
    failure: str | None = None

    def __post_init__(self) -> None:
        if not self.step_id.strip() or not self.description.strip():
            raise ValueError("step_id and description must not be empty")
        if self.cost < 0:
            raise ValueError("step cost must be non-negative")
        if len(set(self.dependencies)) != len(self.dependencies):
            raise ValueError("duplicate dependencies are not allowed")


@dataclass(frozen=True)
class PlanValidation:
    valid: bool
    errors: tuple[str, ...] = ()
    total_cost: float = 0.0


@dataclass(frozen=True)
class PlanAlternative:
    plan_id: str
    steps: tuple[PlanStep, ...]
    total_cost: float
    rationale: str


@dataclass
class AdaptivePlan:
    goal: str
    steps: dict[str, PlanStep] = field(default_factory=dict)
    revision: int = 0
    history: list[str] = field(default_factory=list)

    def add(self, step: PlanStep) -> None:
        if step.step_id in self.steps:
            raise ValueError(f"duplicate step: {step.step_id}")
        self.steps[step.step_id] = step

    def ready_steps(self) -> tuple[PlanStep, ...]:
        completed = {step_id for step_id, step in self.steps.items() if step.status == StepStatus.COMPLETED}
        ready = [
            step for step in self.steps.values()
            if step.status == StepStatus.PENDING and all(dep in completed for dep in step.dependencies)
        ]
        for step in ready:
            step.status = StepStatus.READY
        return tuple(sorted(ready, key=lambda item: (-item.priority, item.cost, item.step_id)))

    def remaining(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.steps.values() if step.status not in {StepStatus.COMPLETED, StepStatus.SKIPPED})

    def completed(self) -> tuple[PlanStep, ...]:
        return tuple(step for step in self.steps.values() if step.status == StepStatus.COMPLETED)


class PlanValidator:
    def validate(self, plan: AdaptivePlan, *, max_cost: float | None = None) -> PlanValidation:
        errors: list[str] = []
        ids = set(plan.steps)
        if not plan.goal.strip():
            errors.append("goal is empty")
        for step in plan.steps.values():
            missing = [dep for dep in step.dependencies if dep not in ids]
            if missing:
                errors.append(f"{step.step_id}: missing dependencies {missing}")
            if step.step_id in step.dependencies:
                errors.append(f"{step.step_id}: self dependency")
        if self._has_cycle(plan):
            errors.append("dependency cycle detected")
        total = sum(step.cost for step in plan.steps.values())
        if max_cost is not None and total > max_cost:
            errors.append(f"plan cost {total:g} exceeds budget {max_cost:g}")
        return PlanValidation(not errors, tuple(errors), total)

    @staticmethod
    def _has_cycle(plan: AdaptivePlan) -> bool:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for dep in plan.steps[node].dependencies:
                if dep in plan.steps and visit(dep):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in plan.steps)


class GoalDecomposer:
    """Convert high-level goal descriptions into a validated plan graph."""

    def decompose(self, goal: str, subtasks: Iterable[str | Mapping[str, object]]) -> AdaptivePlan:
        plan = AdaptivePlan(goal)
        previous: str | None = None
        for index, item in enumerate(subtasks):
            if isinstance(item, Mapping):
                description = str(item.get("description", "")).strip()
                step_id = str(item.get("step_id", f"step-{index}"))
                dependencies = tuple(str(value) for value in item.get("dependencies", ()))
                cost = float(item.get("cost", 1.0))
                priority = int(item.get("priority", 0))
            else:
                description = str(item).strip()
                step_id = f"step-{index}"
                dependencies = (previous,) if previous else ()
                cost = 1.0
                priority = 0
            step = PlanStep(step_id, description, dependencies, cost, priority)
            plan.add(step)
            previous = step_id
        return plan


class PlanOptimizer:
    """Generate deterministic alternative orderings without changing semantics."""

    def alternatives(self, plan: AdaptivePlan, *, count: int = 3) -> tuple[PlanAlternative, ...]:
        if count <= 0:
            raise ValueError("count must be positive")
        validation = PlanValidator().validate(plan)
        if not validation.valid:
            return ()
        alternatives = [
            PlanAlternative("cost-first", tuple(sorted(plan.steps.values(), key=lambda x: (x.cost, x.step_id))), validation.total_cost, "prioritize lower estimated cost"),
            PlanAlternative("priority-first", tuple(sorted(plan.steps.values(), key=lambda x: (-x.priority, x.cost, x.step_id))), validation.total_cost, "prioritize explicit task priority"),
            PlanAlternative("dependency-first", tuple(plan.steps[key] for key in self._topological_order(plan)), validation.total_cost, "preserve deterministic dependency order"),
        ]
        return tuple(alternatives[:count])

    @staticmethod
    def _topological_order(plan: AdaptivePlan) -> list[str]:
        remaining = set(plan.steps)
        order: list[str] = []
        while remaining:
            ready = sorted(node for node in remaining if not (set(plan.steps[node].dependencies) & remaining))
            if not ready:
                raise ValueError("cannot order cyclic plan")
            order.extend(ready)
            remaining.difference_update(ready)
        return order


class PlanReviser:
    """Revise failed plans while preserving completed work."""

    def revise(self, plan: AdaptivePlan, failed_step_id: str, feedback: str, replacement: PlanStep | None = None) -> AdaptivePlan:
        if failed_step_id not in plan.steps:
            raise KeyError(failed_step_id)
        failed = plan.steps[failed_step_id]
        failed.status = StepStatus.FAILED
        failed.failure = str(feedback)
        plan.history.append(f"step {failed_step_id} failed: {feedback}")
        plan.revision += 1
        if replacement is not None:
            replacement.dependencies = tuple(
                dep for dep in replacement.dependencies if dep in plan.steps and dep != failed_step_id
            )
            plan.steps[replacement.step_id] = replacement
            for step in plan.steps.values():
                if failed_step_id in step.dependencies and step.status == StepStatus.PENDING:
                    step.dependencies = tuple(
                        replacement.step_id if dep == failed_step_id else dep for dep in step.dependencies
                    )
        return plan


class AdvancedPlanner:
    """Facade combining decomposition, validation, alternatives and revision."""

    def __init__(self, *, max_cost: float | None = None):
        self.max_cost = max_cost
        self.decomposer = GoalDecomposer()
        self.validator = PlanValidator()
        self.optimizer = PlanOptimizer()
        self.reviser = PlanReviser()

    def create(self, goal: str, subtasks: Iterable[str | Mapping[str, object]]) -> AdaptivePlan:
        plan = self.decomposer.decompose(goal, subtasks)
        validation = self.validator.validate(plan, max_cost=self.max_cost)
        if not validation.valid:
            raise ValueError("invalid plan: " + "; ".join(validation.errors))
        return plan

    def validate(self, plan: AdaptivePlan) -> PlanValidation:
        return self.validator.validate(plan, max_cost=self.max_cost)

    def alternatives(self, plan: AdaptivePlan, count: int = 3) -> tuple[PlanAlternative, ...]:
        return self.optimizer.alternatives(plan, count=count)

    def revise(self, plan: AdaptivePlan, failed_step_id: str, feedback: str, replacement: PlanStep | None = None) -> AdaptivePlan:
        revised = self.reviser.revise(plan, failed_step_id, feedback, replacement)
        validation = self.validator.validate(revised, max_cost=self.max_cost)
        if not validation.valid:
            raise ValueError("revised plan is invalid: " + "; ".join(validation.errors))
        return revised
