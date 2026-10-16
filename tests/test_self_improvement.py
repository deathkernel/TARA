from src.polyglot.benchmark import PolyglotBenchmark
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.result import ExecutionResult
from src.polyglot.self_improvement import SelfImprovementEngine


class FakeSortingExecutor:
    """Deterministic benchmark double; never executes candidate source."""

    def execute(self, candidate, stdin=""):
        if "sorted" not in candidate.source:
            stdout = "wrong\n"
        else:
            values = list(map(int, stdin.split()))
            count = values[0]
            stdout = " ".join(map(str, sorted(values[1:1 + count]))) + "\n"
        return ExecutionResult(
            language=candidate.language,
            success=True,
            phase="fake",
            exit_code=0,
            stdout=stdout,
            stderr="",
            duration_ms=1.0,
        )


def test_engine_uses_benchmark_feedback_and_keeps_verified_candidate():
    calls = []

    def generator(problem, feedback_text, feedback, count):
        calls.append((problem, feedback_text, feedback, count))
        if not feedback:
            yield PolyglotCandidate(problem, "python", "print('wrong')\n")
        else:
            yield PolyglotCandidate(
                problem,
                "python",
                "data=list(map(int,input().split()))\n"
                "n=data[0]\n"
                "print(*sorted(data[1:1+n]))\n",
            )

    benchmark = PolyglotBenchmark(executor=FakeSortingExecutor())
    result = SelfImprovementEngine(benchmark=benchmark).improve(
        "sorting", generator, rounds=2, candidates_per_round=1
    )
    assert result.best is not None
    assert result.best.verified
    assert len(result.history) == 2
    assert len(calls) == 2
    assert calls[1][2]
