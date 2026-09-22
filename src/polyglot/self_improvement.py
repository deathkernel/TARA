"""Iterative candidate improvement driven by executable benchmark feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .benchmark import BenchmarkResult, PolyglotBenchmark, rank_benchmarks
from .candidate import PolyglotCandidate


CandidateGenerator = Callable[[str, str, tuple[str, ...], int], Iterable[PolyglotCandidate]]


@dataclass(frozen=True)
class ImprovementResult:
    best: BenchmarkResult | None
    history: tuple[BenchmarkResult, ...]
    rounds: int


class SelfImprovementEngine:
    """Generate, test, learn from failures, and retain only benchmarked candidates.

    The engine never marks a candidate as correct merely because generation
    succeeded. Verification is always performed by the executable benchmark.
    """

    def __init__(self, benchmark: PolyglotBenchmark | None = None) -> None:
        self.benchmark = benchmark or PolyglotBenchmark()

    def improve(
        self,
        problem: str,
        generator: CandidateGenerator,
        rounds: int = 3,
        candidates_per_round: int = 4,
    ) -> ImprovementResult:
        if rounds < 1 or candidates_per_round < 1:
            raise ValueError("rounds and candidates_per_round must be positive")

        history: list[BenchmarkResult] = []
        feedback: tuple[str, ...] = ()
        best: BenchmarkResult | None = None

        for round_index in range(rounds):
            generated = list(generator(problem, "\n".join(feedback), feedback, candidates_per_round))
            if not generated:
                break
            results = [self.benchmark.run(candidate, problem) for candidate in generated]
            history.extend(results)
            ranked = rank_benchmarks(results)
            round_best = ranked[0]
            if best is None or rank_benchmarks([best, round_best])[0] is round_best:
                best = round_best
            feedback = tuple(message for result in ranked[:2] for message in result.failures)
            if round_best.verified:
                # A fully verified solution is retained immediately. Further
                # rounds may still be useful for performance optimization.
                feedback = (f"Verified candidate runtime: {round_best.total_runtime_ms:.3f} ms",)

        return ImprovementResult(best=best, history=tuple(history), rounds=round_index + 1 if history else 0)
