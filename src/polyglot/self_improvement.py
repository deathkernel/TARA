"""Iterative candidate improvement driven by executable benchmark feedback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from .archive import CandidateArchive
from .benchmark import BenchmarkResult, PolyglotBenchmark, rank_benchmarks
from .candidate import PolyglotCandidate


CandidateGenerator = Callable[[str, str, tuple[str, ...], int], Iterable[PolyglotCandidate]]


@dataclass(frozen=True)
class ImprovementResult:
    best: BenchmarkResult | None
    history: tuple[BenchmarkResult, ...]
    rounds: int


class SelfImprovementEngine:
    """Generate, test, learn from failures, and optionally retain benchmark memory."""

    def __init__(
        self,
        benchmark: PolyglotBenchmark | None = None,
        archive: CandidateArchive | None = None,
    ) -> None:
        self.benchmark = benchmark or PolyglotBenchmark()
        self.archive = archive

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
        completed_rounds = 0

        for _ in range(rounds):
            generated = list(generator(problem, "\n".join(feedback), feedback, candidates_per_round))
            if not generated:
                break
            results = [self.benchmark.run(candidate, problem) for candidate in generated]
            history.extend(results)
            if self.archive is not None:
                for result in results:
                    self.archive.save(result)

            ranked = rank_benchmarks(results)
            round_best = ranked[0]
            if best is None or rank_benchmarks([best, round_best])[0] is round_best:
                best = round_best

            # Preserve concrete failures for the next generation round. Once
            # a candidate verifies, add its runtime as an optimization signal
            # instead of throwing away all information from failed peers.
            failure_feedback = tuple(message for result in ranked[:2] for message in result.failures)
            if round_best.verified:
                feedback = (f"A verified candidate ran in {round_best.total_runtime_ms:.3f} ms; seek a correct candidate with equal or lower runtime.",) + failure_feedback
            else:
                feedback = failure_feedback
            completed_rounds += 1

        return ImprovementResult(best=best, history=tuple(history), rounds=completed_rounds)
