"""Basic persistent task scheduler for TARA."""

from __future__ import annotations

from datetime import datetime, timezone
from .autonomous_orchestrator import AutonomousOrchestrator, TaskNode
from .task_resources import ResourceBudget, ScheduledTask, TaskStore, order_ready


class TaskScheduler:
    """Turn persisted due tasks into bounded orchestrator tasks."""

    def __init__(self, store: TaskStore | None = None, *, budget: ResourceBudget | None = None):
        self.store = store
        self.budget = budget or ResourceBudget()

    def due_tasks(self, now: datetime | None = None) -> list[ScheduledTask]:
        tasks = self.store.load() if self.store else []
        return order_ready(tasks, now or datetime.now(timezone.utc))[: self.budget.max_tasks]

    def run_due(self, executor, *, now: datetime | None = None):
        due = self.due_tasks(now)
        queue = [TaskNode(task.task_id, task.description, priority=task.priority) for task in due]
        orchestrator = AutonomousOrchestrator(max_steps=self.budget.max_tasks)
        orchestrator.queue.extend(queue)
        report = orchestrator.run(executor)
        if self.store:
            remaining = [task for task in self.store.load() if task.task_id not in {item.task_id for item in due}]
            self.store.save(remaining)
        return report
