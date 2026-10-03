from src.polyglot.benchmark import PolyglotBenchmark, rank_benchmarks
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.problems import get_problem, list_problems


def test_problem_registry():
    assert {p.name for p in list_problems()} == {"sorting", "binary_search"}
    assert get_problem("sorting").tests


def test_default_benchmark_reports_disabled_execution():
    candidate = PolyglotCandidate(
        problem="sorting",
        language="python",
        source="print('wrong')\n",
    )
    result = PolyglotBenchmark().run(candidate)
    assert not result.verified
    assert result.passed == 0
    assert result.correctness == 0.0
    assert result.failures
    assert "disabled" in result.failures[0]


def test_benchmark_accepts_injected_execution_backend():
    class FakeExecutor:
        def execute(self, candidate, stdin):
            from src.polyglot.result import ExecutionResult
            spec = get_problem(candidate.problem)
            case = next(case for case in spec.tests if case.stdin == stdin)
            expected = " ".join(case.expected_stdout.split())
            return ExecutionResult(candidate.language, True, "run", 0, expected, "", 1.0, False, None)

    candidate = PolyglotCandidate(
        problem="sorting",
        language="python",
        source="not actually executed",
    )
    result = PolyglotBenchmark(executor=FakeExecutor()).run(candidate)
    assert result.verified
    assert result.passed == result.total
    assert result.correctness == 1.0


def test_rank_prefers_correctness_then_runtime():
    fast_wrong = PolyglotCandidate("sorting", "python", "")
    slow_right = PolyglotCandidate("sorting", "python", "")
    from src.polyglot.benchmark import BenchmarkResult
    results = [
        BenchmarkResult(fast_wrong, 1, 4, 0.25, 1.0, ("failure",)),
        BenchmarkResult(slow_right, 4, 4, 1.0, 100.0, ()),
    ]
    assert rank_benchmarks(results)[0].verified
