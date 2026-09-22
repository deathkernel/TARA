from src.brain import TARABrain
from src.unified_cognitive_loop import ActionOutcome


class DummyModel:
    pass


def test_brain_exposes_unified_cognitive_loop():
    brain = TARABrain(DummyModel())
    report = brain.run_cognitive_loop(
        "demo",
        ["step one"],
        ["observe"],
        action_executor=lambda request: ActionOutcome("done", True, score=1.0),
        expected="done",
    )
    assert report.completed
    assert len(report.traces) == 1


def test_brain_loop_can_be_stopped():
    brain = TARABrain(DummyModel())
    brain.start_cognitive_loop("demo", ["step one"])
    brain.stop_cognitive_loop()
    assert brain.cognitive_loop.stopped
