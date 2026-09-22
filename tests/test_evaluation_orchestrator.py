from pathlib import Path

from src.evaluation_orchestrator import CheckpointEvaluator, EvaluationOrchestrator, EvaluationDecision
from src.intelligence_benchmark import BenchmarkCase, BenchmarkReport


def test_decision_rejects_non_improvement():
    evaluator = object.__new__(CheckpointEvaluator)
    evaluator.gate = type("Gate", (), {"check": lambda self, a, b: False})()
    base = type("Eval", (), {"benchmark": type("B", (), {"overall_score": 0.5})(), "fingerprint": "a"})()
    candidate = type("Eval", (), {"benchmark": type("B", (), {"overall_score": 0.5})(), "fingerprint": "b"})()
    decision = evaluator.compare(base, candidate)
    assert decision.accepted is False
    assert "strictly improve" in decision.reason


def test_decision_accepts_strict_improvement_without_regression():
    evaluator = object.__new__(CheckpointEvaluator)
    evaluator.gate = type("Gate", (), {"check": lambda self, a, b: False})()
    base = type("Eval", (), {"benchmark": type("B", (), {"overall_score": 0.5})(), "fingerprint": "a"})()
    candidate = type("Eval", (), {"benchmark": type("B", (), {"overall_score": 0.75})(), "fingerprint": "b"})()
    decision = evaluator.compare(base, candidate)
    assert decision.accepted is True


def test_orchestrator_without_improvement_is_measurement_only(tmp_path: Path):
    orchestrator = EvaluationOrchestrator()
    orchestrator.evaluator = type("Evaluator", (), {
        "evaluate": lambda self, path, name: type("Eval", (), {
            "benchmark": type("B", (), {"overall_score": 0.25})(), "fingerprint": "fp"
        })()
    })()
    cycle = orchestrator.run("baseline.pt")
    assert cycle.candidate is None
    assert cycle.decision.accepted is False
    assert "no improvement" in cycle.decision.reason
