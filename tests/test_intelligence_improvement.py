from src.intelligence_benchmark import BenchmarkCase, IntelligenceBenchmark, normalized_text_score
from src.intelligence_improvement import FailureAnalyzer, ImprovementPlanner, IntelligenceImprovementEngine


def _benchmark():
    return IntelligenceBenchmark("improvement", (
        BenchmarkCase("code", "coding", "code", "ok", normalized_text_score),
        BenchmarkCase("logic", "reasoning", "logic", "ok", normalized_text_score),
        BenchmarkCase("memory", "memory", "memory", "ok", normalized_text_score),
    ))


def test_failure_analysis_and_targeting_are_deterministic():
    benchmark = _benchmark()
    baseline = benchmark.run(lambda prompt: "wrong" if prompt == "logic" else "ok")
    failures = FailureAnalyzer().analyze(baseline)
    proposal = ImprovementPlanner().propose(failures)
    assert failures[0].case_id == "logic"
    assert proposal.targets[0].category == "reasoning"
    assert proposal.proposal_id == proposal.fingerprint[:16]


def test_engine_accepts_only_strict_improvement_without_regression():
    benchmark = _benchmark()
    baseline = benchmark.run(lambda prompt: "wrong" if prompt == "logic" else "ok")
    candidate = benchmark.run(lambda _: "ok")
    report = IntelligenceImprovementEngine().improve(baseline, lambda _: candidate)
    assert report.decision.accepted
    assert report.decision.regression.passed
    assert report.decision.candidate_score > report.decision.baseline_score


def test_engine_rejects_candidate_with_category_regression():
    benchmark = _benchmark()
    baseline = benchmark.run(lambda prompt: "wrong" if prompt == "coding" else "ok")
    candidate = benchmark.run(lambda prompt: "wrong" if prompt in {"coding", "memory"} else "ok")
    report = IntelligenceImprovementEngine().improve(baseline, lambda _: candidate)
    assert not report.decision.accepted
    assert "memory" in report.decision.regression.regressions


def test_engine_does_not_train_when_baseline_is_clean():
    benchmark = _benchmark()
    baseline = benchmark.run(lambda _: "ok")
    called = False

    def runner(_):
        nonlocal called
        called = True
        return baseline

    report = IntelligenceImprovementEngine().improve(baseline, runner)
    assert not called
    assert not report.decision.accepted
    assert report.candidate.fingerprint == baseline.fingerprint
