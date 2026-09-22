from src.self_improvement_lab import FailureAnalyzer, PromotionEngine, RegressionGate, SelfImprovementLab
from src.polyglot.benchmark import BenchmarkResult
from src.polyglot.candidate import PolyglotCandidate


def result(source, correctness, runtime, failures=()):
    candidate = PolyglotCandidate("sorting", "python", source)
    total = 2
    return BenchmarkResult(candidate, round(correctness * total), total, correctness, runtime, tuple(failures))


def test_failure_analyzer_extracts_targeted_signal():
    analysis = FailureAnalyzer().analyze(result("x", 0.5, 10, ("case 2: timed out",)))
    assert analysis.category == "timeout"
    assert "reduce complexity" in analysis.suggested_focus


def test_regression_gate_blocks_correctness_regression():
    baseline = result("old", 1.0, 20)
    worse = result("new", 0.5, 2)
    assert not RegressionGate().accept(baseline, worse)


def test_promotion_requires_measured_improvement():
    baseline = result("old algorithm", 1.0, 100)
    faster = result("new", 1.0, 5)
    decision = PromotionEngine().decide(baseline, (faster,))
    assert decision.promoted


def test_lab_runs_mutations_only_through_benchmark():
    baseline = result("old", 1.0, 100)

    def mutator(parent, analysis, count):
        yield PolyglotCandidate(parent.problem, parent.language, "new")

    lab = SelfImprovementLab(mutator=mutator)
    cycle = lab.cycle(baseline, "sorting", mutation_count=1)
    assert len(cycle.mutations) == 1
    assert len(cycle.experiments) == 1
