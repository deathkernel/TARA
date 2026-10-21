from pathlib import Path
from types import SimpleNamespace

from src.evaluation_orchestrator import CheckpointEvaluator, EvaluationOrchestrator
from src.intelligence_benchmark import (
    BenchmarkReport,
    CategoryScore,
    CaseResult,
)


def _evaluation(score: float, *, category_score: float | None = None, fingerprint: str):
    category_score = score if category_score is None else category_score
    report = BenchmarkReport(
        name="tara-capability-v1",
        case_count=1,
        passed_cases=int(score >= 1.0),
        overall_score=score,
        categories=(CategoryScore("coding", category_score, 1, int(category_score >= 1.0)),),
        results=(CaseResult("case-1", "coding", score, score >= 1.0, 1.0),),
        fingerprint=f"benchmark-{fingerprint}",
    )
    return SimpleNamespace(benchmark=report, fingerprint=fingerprint)


def test_decision_rejects_non_improvement():
    evaluator = CheckpointEvaluator()
    baseline = _evaluation(0.5, fingerprint="baseline")
    candidate = _evaluation(0.5, fingerprint="candidate")

    decision = evaluator.compare(baseline, candidate)

    assert decision.accepted is False
    assert "strictly improve" in decision.reason
    assert decision.regression is False


def test_decision_accepts_strict_improvement_without_regression():
    evaluator = CheckpointEvaluator()
    baseline = _evaluation(0.5, fingerprint="baseline")
    candidate = _evaluation(0.75, fingerprint="candidate")

    decision = evaluator.compare(baseline, candidate)

    assert decision.accepted is True
    assert decision.regression is False
    assert decision.candidate_score > decision.baseline_score


def test_decision_rejects_category_regression_even_when_overall_improves():
    evaluator = CheckpointEvaluator(category_tolerance=0.05)
    baseline = _evaluation(0.5, category_score=0.8, fingerprint="baseline")
    candidate = _evaluation(0.75, category_score=0.6, fingerprint="candidate")

    decision = evaluator.compare(baseline, candidate)

    assert decision.accepted is False
    assert decision.regression is True
    assert "regression gate" in decision.reason


def test_orchestrator_without_improvement_is_measurement_only(tmp_path: Path):
    orchestrator = EvaluationOrchestrator()
    orchestrator.evaluator = type(
        "Evaluator",
        (),
        {
            "evaluate": lambda self, path, name=None: _evaluation(0.25, fingerprint="fp"),
        },
    )()

    cycle = orchestrator.run("baseline.pt")

    assert cycle.candidate is None
    assert cycle.decision.accepted is False
    assert "no improvement" in cycle.decision.reason
