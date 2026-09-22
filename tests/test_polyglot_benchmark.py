from src.polyglot.benchmark import PolyglotBenchmark, rank_benchmarks
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.problems import get_problem, list_problems


def test_problem_registry():
    assert {p.name for p in list_problems()} == {"sorting", "binary_search"}
    assert get_problem("sorting").tests


def test_sorting_benchmark_accepts_correct_candidate():
    candidate = PolyglotCandidate(
        problem="sorting",
        language="python",
        source=(
            "data=list(map(int,input().split()))\n"
            "n=data[0]\n"
            "a=data[1:1+n]\n"
            "print(*sorted(a))\n"
        ),
    )
    result = PolyglotBenchmark().run(candidate)
    assert result.verified
    assert result.passed == result.total
    assert result.correctness == 1.0


def test_benchmark_reports_failure():
    candidate = PolyglotCandidate(
        problem="sorting",
        language="python",
        source="print('wrong')\n",
    )
    result = PolyglotBenchmark().run(candidate)
    assert not result.verified
    assert result.failures


def test_rank_prefers_correctness_then_runtime():
    fast_wrong = PolyglotCandidate("sorting", "python", "")
    slow_right = PolyglotCandidate("sorting", "python", "")
    from src.polyglot.benchmark import BenchmarkResult
    results = [
        BenchmarkResult(fast_wrong, 1, 4, 0.25, 1.0, ("failure",)),
        BenchmarkResult(slow_right, 4, 4, 1.0, 100.0, ()),
    ]
    assert rank_benchmarks(results)[0].verified
