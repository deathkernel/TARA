"""Basic resource and persistent scheduling support for TARA Phase 19."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResourceBudget:
    """Small explicit budget used to prevent runaway task execution."""

    max_tasks: int = 64
    max_retries: int = 16

    def __post_init__(self) -> None:
        if self.max_tasks <= 0 or self.max_retries < 0:
            raise ValueError("resource budgets must be positive/non-negative")


@dataclass
class ScheduledTask:
    """Serializable task metadata for a basic persistent scheduler."""

    task_id: str
    description: str
    run_at: str | None = None
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.description.strip():
            raise ValueError("task_id and description must be non-empty")
        if self.priority < 0:
            raise ValueError("priority must be non-negative")

    def due(self, now: datetime | None = None) -> bool:
        if not self.run_at:
            return True
        value = datetime.fromisoformat(self.run_at.replace("Z", "+00:00"))
        current = now or datetime.now(timezone.utc)
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return value <= current


class TaskStore:
    """Minimal JSONL store for restart-safe task metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def save(self, tasks: list[ScheduledTask]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for task in tasks:
                handle.write(json.dumps({"task_id": task.task_id, "description": task.description,
                                         "run_at": task.run_at, "priority": task.priority,
                                         "metadata": task.metadata}, sort_keys=True) + "\n")

    def load(self) -> list[ScheduledTask]:
        if not self.path.exists():
            return []
        tasks: list[ScheduledTask] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                data = json.loads(line)
                tasks.append(ScheduledTask(**data))
        return tasks


def order_ready(tasks: list[ScheduledTask], now: datetime | None = None) -> list[ScheduledTask]:
    """Return due tasks in deterministic priority order."""
    return sorted((task for task in tasks if task.due(now)), key=lambda item: (-item.priority, item.task_id))
