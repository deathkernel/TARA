from src.architecture_optimization import ArchitectureComponent, ArchitectureMetrics, ArchitectureOptimizer, ArchitectureVariant
from src.brain import TARABrain
from src.experience_learning import Experience
from src.tara_core import CoreConfig, CoreStatus, TARACore
from src.unified_cognitive_loop import ActionOutcome


class DummyModel:
    pass


def test_core_health_and_end_to_end_cycle():
    core = TARACore(TARABrain(DummyModel()), config=CoreConfig(seed=3, max_cycles=4))
    health = core.health()
    assert health.healthy
    report = core.run(
        "demo",
        ["step one", "step two"],
        ["perception one", "perception two"],
        action_executor=lambda request: ActionOutcome(request.step.description, True, score=1.0),
        expected=lambda value: isinstance(value, str),
    )
    assert report.completed
    assert core.status is CoreStatus.READY
    assert core.snapshot().cognitive_cycles == 2
    assert core.events()


def test_core_stop_resume():
    core = TARACore(TARABrain(DummyModel()))
    core.stop()
    assert core.status is CoreStatus.STOPPED
    try:
        core.run("x", ["step"], ["p"], action_executor=lambda request: "ok")
    except RuntimeError:
        pass
    else:
        raise AssertionError("stopped core should reject work")
    core.resume()
    assert core.status is CoreStatus.READY


def test_core_learning_and_manifest(tmp_path):
    core = TARACore(TARABrain(DummyModel()))
    report = core.learn([Experience("e1", "s", "a", "ok", 1.0, True)])
    assert report.updates
    path = tmp_path / "runtime.json"
    snapshot = core.save_runtime_manifest(path)
    assert snapshot.fingerprint
    assert path.read_text(encoding="utf-8")


def test_architecture_api_requires_evaluator():
    core = TARACore(TARABrain(DummyModel()))
    variant = ArchitectureVariant("base", (ArchitectureComponent("reasoner", critical=True),))
    try:
        core.optimize_architecture(variant)
    except ValueError as exc:
        assert "ArchitectureOptimizer" in str(exc)
    else:
        raise AssertionError("expected evaluator boundary")


def test_core_replay_boundary():
    core = TARACore(TARABrain(DummyModel()))
    batch = core.build_replay([
        {"text": "verified example", "domain": "core", "verified": True},
        {"text": "ignored example", "domain": "core", "verified": False},
    ])
    assert len(batch.examples) == 1
    assert batch.examples[0].verified is True
