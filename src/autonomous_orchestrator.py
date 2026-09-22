"""Basic autonomous task orchestration for TARA Phase 18.

The orchestrator manages explicit tasks, dependencies, priorities, bounded
retries, verification feedback, and emergency stop/resume. It never invents
or obtains new permissions: external work is performed only by a supplied
callable or an already permissioned ToolController.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class TaskNode:
    """One bounded unit in an orchestration graph."""

    task_id: str
    description: str
    priority: int = 0
    dependencies: tuple[str, ...] = ()
    retries: int = 0
    status: str = "pending"
    result: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.description.strip():
            raise ValueError("task_id and description must be non-empty")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")
        if self.retries < 0:
            raise ValueError("retries must be non-negative")


@dataclass(frozen=True)
class OrchestrationEvent:
    task_id: str
    status: str
    result: Any = None
    error: str | None = None


@dataclass(frozen=True)
class OrchestrationReport:
    completed: int
    failed: int
    pending: int
    stopped: bool
    events: tuple[OrchestrationEvent, ...]


class TaskQueue:
    """Deterministic priority queue with dependency-aware readiness."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskNode] = {}

    def add(self, task: TaskNode) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task id: {task.task_id}")
        self._tasks[task.task_id] = task

    def extend(self, tasks: Iterable[TaskNode]) -> None:
        for task in tasks:
            self.add(task)

    def get(self, task_id: str) -> TaskNode:
        return self._tasks[task_id]

    def ready(self) -> list[TaskNode]:
        ready = []
        for task in self._tasks.values():
            if task.status != "pending":
                continue
            if all(self._tasks.get(dep) is not None and self._tasks[dep].status == "completed"
                   for dep in task.dependencies):
                ready.append(task)
        return sorted(ready, key=lambda item: (-item.priority, item.task_id))

    def all(self) -> tuple[TaskNode, ...]:
        return tuple(self._tasks.values())


class AutonomousOrchestrator:
    """Run a finite task graph through a host-supplied executor."""

    def __init__(self, queue: TaskQueue | None = None, *, max_steps: int = 64) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.queue = queue or TaskQueue()
        self.max_steps = max_steps
        self._stopped = False
        self.events: list[OrchestrationEvent] = []

    def stop(self) -> None:
        self._stopped = True

    def resume(self) -> None:
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def _blocked_dependencies(self) -> bool:
        for task in self.queue.all():
            if task.status != "pending":
                continue
            if any(dep not in {item.task_id for item in self.queue.all()} or
                   self.queue.get(dep).status == "failed" for dep in task.dependencies):
                task.status = "failed"
                task.error = "dependency failed or is missing"
                self.events.append(OrchestrationEvent(task.task_id, "failed", error=task.error))
                return True
        return False

    def run(self, executor: Callable[[TaskNode], Any]) -> OrchestrationReport:
        if not callable(executor):
            raise TypeError("executor must be callable")

        steps = 0
        while steps < self.max_steps and not self._stopped:
            ready = self.queue.ready()
            if not ready:
                self._blocked_dependencies()
                if not self.queue.ready():
                    break
                continue

            task = ready[0]
            task.status = "running"
            try:
                task.result = executor(task)
                task.status = "completed"
                task.error = None
                self.events.append(OrchestrationEvent(task.task_id, "completed", task.result))
            except Exception as exc:
                task.error = f"{type(exc).__name__}: {exc}"
                if task.retries > 0:
                    task.retries -= 1
                    task.status = "pending"
                    self.events.append(OrchestrationEvent(task.task_id, "retry", error=task.error))
                else:
                    task.status = "failed"
                    self.events.append(OrchestrationEvent(task.task_id, "failed", error=task.error))
            steps += 1

        completed = sum(task.status == "completed" for task in self.queue.all())
        failed = sum(task.status == "failed" for task in self.queue.all())
        pending = sum(task.status in {"pending", "running"} for task in self.queue.all())
        return OrchestrationReport(completed, failed, pending, self._stopped, tuple(self.events))

    def requeue_failed(self, *, retries: int = 1) -> int:
        if retries < 0:
            raise ValueError("retries must be non-negative")
        count = 0
        for task in self.queue.all():
            if task.status == "failed":
                task.status = "pending"
                task.retries = retries
                task.error = None
                count += 1
        return count
