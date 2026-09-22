from src.intelligence_benchmark import (
    BenchmarkCase,
    IntelligenceBenchmark,
    benchmark_cases,
    compare_regression,
    normalized_text_score,
)


def test_benchmark_produces_deterministic_categories_and_fingerprint():
    benchmark = IntelligenceBenchmark("smoke", benchmark_cases())

    answers = {
        "return 2 + 2": "4",
        "Is 7 greater than 3?": True,
        "Recall token TARA-37": "TARA-37",
        "What comes first: plan or execute?": "plan",
    }
    report = benchmark.run(lambda prompt: answers[prompt])

    assert report.passed
    assert report.overall_score == 1.0
    assert [item.category for item in report.categories] == ["coding", "memory", "planning", "reasoning"]
    assert len(report.fingerprint) == 64


def test_benchmark_records_solver_failure_without_crashing():
    case = BenchmarkCase("x", "coding", "x", "ok", normalized_text_score)
    report = IntelligenceBenchmark("failure", [case]).run(lambda _: (_ for _ in ()).throw(RuntimeError("boom")))

    assert not report.passed
    assert report.overall_score == 0.0
    assert report.results[0].error == "boom"


def test_regression_gate_detects_category_drop():
    cases = (
        BenchmarkCase("a", "coding", "a", "ok", normalized_text_score),
        BenchmarkCase("b", "reasoning", "b", "ok", normalized_text_score),
    )
    benchmark = IntelligenceBenchmark("gate", cases)
    baseline = benchmark.run(lambda _: "ok")
    candidate = benchmark.run(lambda prompt: "wrong" if prompt == "b" else "ok")

    gate = compare_regression(baseline, candidate, maximum_category_drop=0.05)
    assert not gate.passed
    assert gate.regressions == ("reasoning",)
    assert gate.overall_delta == -0.5
