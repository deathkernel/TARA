from datetime import datetime, timezone

from src.task_resources import ResourceBudget, ScheduledTask, TaskStore, order_ready
from src.task_scheduler import TaskScheduler


def test_due_tasks_are_priority_ordered():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tasks = [
        ScheduledTask("low", "low", priority=1),
        ScheduledTask("high", "high", priority=5),
        ScheduledTask("future", "future", run_at="2027-01-01T00:00:00+00:00", priority=9),
    ]
    assert [task.task_id for task in order_ready(tasks, now)] == ["high", "low"]


def test_store_round_trip(tmp_path):
    store = TaskStore(tmp_path / "tasks.jsonl")
    original = [ScheduledTask("a", "do a", priority=2, metadata={"kind": "demo"})]
    store.save(original)
    loaded = store.load()
    assert loaded == original


def test_scheduler_respects_budget(tmp_path):
    store = TaskStore(tmp_path / "tasks.jsonl")
    store.save([ScheduledTask(str(i), f"task {i}") for i in range(5)])
    scheduler = TaskScheduler(store, budget=ResourceBudget(max_tasks=2))
    report = scheduler.run_due(lambda task: task.task_id)
    assert report.completed == 2
    assert len(store.load()) == 3


def test_scheduler_retains_failed_due_task_and_applies_retries(tmp_path):
    store = TaskStore(tmp_path / "tasks.jsonl")
    store.save([ScheduledTask("retry", "retry me", metadata={"retries": 2})])
    scheduler = TaskScheduler(store, budget=ResourceBudget(max_tasks=1, max_retries=1))
    report = scheduler.run_due(lambda task: (_ for _ in ()).throw(RuntimeError("boom")))
    assert report.failed == 1
    remaining = store.load()
    assert len(remaining) == 1
