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
        queue = []
        for task in due:
            requested_retries = int(task.metadata.get("retries", 0)) if isinstance(task.metadata, dict) else 0
            retries = min(max(0, requested_retries), self.budget.max_retries)
            queue.append(TaskNode(task.task_id, task.description, priority=task.priority, retries=retries))
        orchestrator = AutonomousOrchestrator(max_steps=self.budget.max_tasks)
        orchestrator.queue.extend(queue)
        report = orchestrator.run(executor)
        if self.store:
            completed_ids = {
                event.task_id for event in report.events if event.status == "completed"
            }
            remaining = [
                task for task in self.store.load()
                if task.task_id not in completed_ids
            ]
            self.store.save(remaining)
        return report
