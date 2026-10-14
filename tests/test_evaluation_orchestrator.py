from pathlib import Path

from src.evaluation_orchestrator import CheckpointEvaluator, EvaluationOrchestrator
from src.intelligence_benchmark import BenchmarkCase, IntelligenceBenchmark, normalized_text_score
from src.model_capability_runner import ModelEvaluation


def _evaluation(outputs: dict[str, str], fingerprint: str) -> ModelEvaluation:
    cases = (
        BenchmarkCase("coding", "coding", "code", "ok", normalized_text_score),
        BenchmarkCase("reasoning", "reasoning", "logic", "ok", normalized_text_score),
    )
    benchmark = IntelligenceBenchmark("tara-capability-v1", cases).run(lambda prompt: outputs[prompt])
    return ModelEvaluation(fingerprint, benchmark, fingerprint, fingerprint)


def test_decision_rejects_non_improvement():
    evaluator = CheckpointEvaluator()
    base = _evaluation({"code": "ok", "logic": "bad"}, "a")
    candidate = _evaluation({"code": "ok", "logic": "bad"}, "b")
    decision = evaluator.compare(base, candidate)
    assert decision.accepted is False
    assert "strictly improve" in decision.reason


def test_decision_accepts_strict_improvement_without_regression():
    evaluator = CheckpointEvaluator()
    base = _evaluation({"code": "bad", "logic": "bad"}, "a")
    candidate = _evaluation({"code": "ok", "logic": "bad"}, "b")
    decision = evaluator.compare(base, candidate)
    assert decision.accepted is True
    assert decision.regression is False


def test_orchestrator_without_improvement_is_measurement_only(tmp_path: Path):
    orchestrator = EvaluationOrchestrator()
    orchestrator.evaluator = type("Evaluator", (), {
        "evaluate": lambda self, path, name=None: _evaluation({"code": "bad", "logic": "bad"}, "fp")
    })()
    cycle = orchestrator.run("baseline.pt")
    assert cycle.candidate is None
    assert cycle.decision.accepted is False
    assert "no improvement" in cycle.decision.reason
