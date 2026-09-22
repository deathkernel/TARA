from src.autonomous_orchestrator import AutonomousOrchestrator, TaskNode, TaskQueue


def test_queue_respects_dependencies_and_priority():
    queue = TaskQueue()
    queue.add(TaskNode("b", "second", priority=10, dependencies=("a",)))
    queue.add(TaskNode("a", "first", priority=1))
    assert [task.task_id for task in queue.ready()] == ["a"]
    queue.get("a").status = "completed"
    assert [task.task_id for task in queue.ready()] == ["b"]


def test_orchestrator_runs_and_retries():
    queue = TaskQueue()
    queue.add(TaskNode("a", "first", retries=1))
    calls = []

    def execute(task):
        calls.append(task.task_id)
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return "ok"

    report = AutonomousOrchestrator(queue).run(execute)
    assert report.completed == 1
    assert report.failed == 0
    assert calls == ["a", "a"]


def test_stop_prevents_execution():
    queue = TaskQueue()
    queue.add(TaskNode("a", "first"))
    orchestrator = AutonomousOrchestrator(queue)
    orchestrator.stop()
    report = orchestrator.run(lambda task: "never")
    assert report.stopped is True
    assert report.pending == 1


def test_requeue_failed_tasks():
    queue = TaskQueue()
    task = TaskNode("a", "first")
    queue.add(task)
    task.status = "failed"
    orchestrator = AutonomousOrchestrator(queue)
    assert orchestrator.requeue_failed(retries=2) == 1
    assert task.status == "pending"
    assert task.retries == 2
