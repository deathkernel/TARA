"""Benchmark verified polyglot candidates using deterministic problem cases."""

from __future__ import annotations

from dataclasses import dataclass

from .candidate import PolyglotCandidate
from .execution import PolyglotExecutor
from .problems import ProblemSpec, get_problem


@dataclass(frozen=True)
class BenchmarkResult:
    candidate: PolyglotCandidate
    passed: int
    total: int
    correctness: float
    total_runtime_ms: float
    failures: tuple[str, ...]

    @property
    def verified(self) -> bool:
        return self.passed == self.total


class PolyglotBenchmark:
    def __init__(self, executor: PolyglotExecutor | None = None) -> None:
        self.executor = executor or PolyglotExecutor()

    def run(self, candidate: PolyglotCandidate, problem: ProblemSpec | str | None = None) -> BenchmarkResult:
        spec = get_problem(problem) if isinstance(problem, str) else (problem or get_problem(candidate.problem))
        passed = 0
        runtime = 0.0
        failures: list[str] = []
        for index, case in enumerate(spec.tests, start=1):
            result = self.executor.execute(candidate, case.stdin)
            runtime += result.duration_ms
            actual = " ".join(result.stdout.split())
            expected = " ".join(case.expected_stdout.split())
            if result.success and actual == expected:
                passed += 1
            else:
                detail = result.stderr.strip() or result.error or "output mismatch"
                failures.append(f"case {index}: {detail}; expected={expected!r}; actual={actual!r}")
        correctness = passed / len(spec.tests) if spec.tests else 0.0
        return BenchmarkResult(candidate, passed, len(spec.tests), correctness, runtime, tuple(failures))


def rank_benchmarks(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Rank by correctness first, then lower runtime among equally correct candidates."""
    return sorted(results, key=lambda x: (-x.correctness, x.total_runtime_ms))
