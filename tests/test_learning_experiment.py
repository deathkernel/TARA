from pathlib import Path

from src.intelligence_benchmark import BenchmarkCase, IntelligenceBenchmark, normalized_text_score
from src.learning_experiment import LearningExperiment
from src.model_capability_runner import ModelEvaluation


def _evaluation(score: float, fingerprint: str) -> ModelEvaluation:
    cases = (BenchmarkCase("x", "coding", "x", "ok", normalized_text_score),)
    benchmark = IntelligenceBenchmark("tara-capability-v1", cases).run(lambda _: "ok" if score else "bad")
    return ModelEvaluation(fingerprint, benchmark, fingerprint, fingerprint)


def test_learning_experiment_rejects_non_improving_candidate(tmp_path: Path):
    experiment = LearningExperiment()
    baseline = _evaluation(1.0, "base")
    candidate = _evaluation(1.0, "candidate")
    experiment.evaluator.evaluate = lambda path: baseline if str(path) == "base" else candidate
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"checkpoint")
    report = experiment.run("base", train_candidate=lambda _: candidate_path)
    assert not report.decision.accepted
    assert report.decision.baseline_score == report.decision.candidate_score


def test_learning_experiment_reports_category_delta(tmp_path: Path):
    experiment = LearningExperiment()
    cases = (
        BenchmarkCase("a", "coding", "a", "ok", normalized_text_score),
        BenchmarkCase("b", "reasoning", "b", "ok", normalized_text_score),
    )
    def evaluation(value: str) -> ModelEvaluation:
        outputs = {"base": {"a": "ok", "b": "bad"}, "candidate": {"a": "ok", "b": "ok"}}[value]
        report = IntelligenceBenchmark("tara-capability-v1", cases).run(lambda prompt: outputs[prompt])
        return ModelEvaluation(value, report, value, value)
    experiment.evaluator.evaluate = evaluation
    candidate_path = tmp_path / "candidate.pt"
    candidate_path.write_bytes(b"checkpoint")
    result = experiment.run("base", train_candidate=lambda _: candidate_path)
    assert result.decision.accepted
    assert {item.category: item.delta for item in result.capability_deltas}["reasoning"] == 1.0
