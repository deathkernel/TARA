from src.polyglot.benchmark import PolyglotBenchmark
from src.polyglot.candidate import PolyglotCandidate
from src.polyglot.problems import get_problem
from src.polyglot.result import ExecutionResult
from src.polyglot.self_improvement import SelfImprovementEngine


class FakeSortingExecutor:
    """Deterministic benchmark executor; never starts a host process."""

    def execute(self, candidate, stdin=""):
        if "sorted" in candidate.source:
            spec = get_problem(candidate.problem)
            stdout = next((case.expected_stdout for case in spec.tests if case.stdin == stdin), "")
            return ExecutionResult(candidate.language, True, "fake", 0, stdout, "", 1.0)
        return ExecutionResult(candidate.language, False, "fake", 1, "wrong\n", "", 1.0)


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

    result = SelfImprovementEngine(benchmark=PolyglotBenchmark(executor=FakeSortingExecutor())).improve("sorting", generator, rounds=2, candidates_per_round=1)
    assert result.best is not None
    assert result.best.verified
    assert len(result.history) == 2
    assert len(calls) == 2
    assert calls[1][2]
