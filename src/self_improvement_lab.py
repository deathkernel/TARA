"""Advanced self-improvement loop for TARA.

The loop treats improvement as a measured experiment:

failure analysis -> targeted mutation/generation -> regression benchmark ->
Pareto/utility comparison -> promotion gate -> experience record.

Generated programs are never executed directly here; all execution remains
behind the existing PolyglotBenchmark/PolyglotExecutor boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Callable, Iterable, Sequence

from .polyglot.archive import CandidateArchive
from .polyglot.benchmark import BenchmarkResult, PolyglotBenchmark
from .polyglot.candidate import PolyglotCandidate


@dataclass(frozen=True)
class FailureAnalysis:
    category: str
    severity: float
    signals: tuple[str, ...]
    suggested_focus: tuple[str, ...]


@dataclass(frozen=True)
class Mutation:
    mutation_id: str
    parent: PolyglotCandidate
    candidate: PolyglotCandidate
    rationale: str


@dataclass(frozen=True)
class ImprovementScore:
    correctness: float
    speed: float
    compactness: float
    regression_safety: float
    total: float


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    score: ImprovementScore
    selected_fingerprint: str | None = None


@dataclass(frozen=True)
class ImprovementCycle:
    baseline: BenchmarkResult
    experiments: tuple[BenchmarkResult, ...]
    analyses: tuple[FailureAnalysis, ...]
    mutations: tuple[Mutation, ...]
    decision: PromotionDecision


class FailureAnalyzer:
    """Convert benchmark failures into structured optimization signals."""

    _rules = (
        ("timeout", ("timed out", "timeout", "time limit"), ("reduce complexity", "avoid repeated scans")),
        ("compile", ("compile", "syntax", "javac", "rustc", "g++"), ("repair syntax", "check language constraints")),
        ("runtime", ("exception", "traceback", "panic", "segmentation", "runtime"), ("guard edge cases", "validate bounds")),
        ("incorrect", ("expected=", "output mismatch"), ("inspect invariant", "expand edge-case handling")),
    )

    def analyze(self, result: BenchmarkResult) -> FailureAnalysis:
        if result.verified:
            return FailureAnalysis("none", 0.0, (), ("optimize runtime without changing semantics",))
        joined = " ".join(result.failures).lower()
        for category, needles, focus in self._rules:
            if any(needle in joined for needle in needles):
                severity = 1.0 - result.correctness
                return FailureAnalysis(category, severity, tuple(result.failures[:4]), focus)
        return FailureAnalysis("unknown", 1.0 - result.correctness, tuple(result.failures[:4]), ("increase test coverage", "inspect assumptions"))


class MutationEngine:
    """Generate targeted variants from a parent candidate.

    The mutation policy is injectable so an LM, evolutionary operator, or
    program-synthesis model can be plugged in without weakening verification.
    """

    def __init__(self, mutator: Callable[[PolyglotCandidate, FailureAnalysis, int], Iterable[PolyglotCandidate]] | None = None) -> None:
        self.mutator = mutator

    def generate(self, parent: PolyglotCandidate, analysis: FailureAnalysis, count: int = 4) -> tuple[Mutation, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        if self.mutator is None:
            return ()
        output: list[Mutation] = []
        seen: set[tuple[str, str]] = set()
        for index, candidate in enumerate(self.mutator(parent, analysis, count)):
            key = (candidate.language, candidate.source.strip())
            if key in seen or key == (parent.language, parent.source.strip()):
                continue
            seen.add(key)
            output.append(Mutation(
                mutation_id=f"m{index:04d}",
                parent=parent,
                candidate=candidate,
                rationale="; ".join(analysis.suggested_focus),
            ))
            if len(output) >= count:
                break
        return tuple(output)


class RegressionGate:
    """Require correctness preservation before accepting an optimization."""

    def __init__(self, minimum_correctness: float = 1.0) -> None:
        if not 0.0 <= minimum_correctness <= 1.0:
            raise ValueError("minimum_correctness must be in [0, 1]")
        self.minimum_correctness = minimum_correctness

    def accept(self, baseline: BenchmarkResult, candidate: BenchmarkResult) -> bool:
        return candidate.correctness >= self.minimum_correctness and candidate.correctness >= baseline.correctness


class MultiObjectiveImprovement:
    """Score candidates using correctness, speed, compactness and regression safety."""

    def score(self, baseline: BenchmarkResult, candidate: BenchmarkResult) -> ImprovementScore:
        correctness = candidate.correctness
        speed = 1.0 if candidate.total_runtime_ms <= 0 else 1.0 / (1.0 + log1p(candidate.total_runtime_ms))
        baseline_len = max(1, len(baseline.candidate.source))
        candidate_len = max(1, len(candidate.candidate.source))
        compactness = min(1.0, baseline_len / candidate_len)
        regression = 1.0 if candidate.correctness >= baseline.correctness else 0.0
        total = 0.60 * correctness + 0.20 * speed + 0.10 * compactness + 0.10 * regression
        return ImprovementScore(correctness, speed, compactness, regression, total)


class PromotionEngine:
    """Promote only candidates that pass a deterministic regression gate."""

    def __init__(self, gate: RegressionGate | None = None, scorer: MultiObjectiveImprovement | None = None) -> None:
        self.gate = gate or RegressionGate()
        self.scorer = scorer or MultiObjectiveImprovement()

    def decide(self, baseline: BenchmarkResult, experiments: Sequence[BenchmarkResult]) -> PromotionDecision:
        eligible = [item for item in experiments if self.gate.accept(baseline, item)]
        baseline_score = self.scorer.score(baseline, baseline)
        if not eligible:
            return PromotionDecision(False, "no experiment passed the regression gate", baseline_score)
        best = max(eligible, key=lambda item: self.scorer.score(baseline, item).total)
        score = self.scorer.score(baseline, best)
        if score.total <= baseline_score.total:
            return PromotionDecision(False, "no candidate improved the measured objective", score)
        return PromotionDecision(True, "candidate passed regression and improved multi-objective utility", score, self._fingerprint(best))

    @staticmethod
    def _fingerprint(result: BenchmarkResult) -> str:
        import hashlib
        return hashlib.sha256(f"{result.candidate.language}\n{result.candidate.source}".encode()).hexdigest()

    def select_promoted(self, baseline: BenchmarkResult, experiments: Sequence[BenchmarkResult]) -> BenchmarkResult | None:
        eligible = [item for item in experiments if self.gate.accept(baseline, item)]
        if not eligible:
            return None
        baseline_score = self.scorer.score(baseline, baseline).total
        best = max(eligible, key=lambda item: self.scorer.score(baseline, item).total)
        return best if self.scorer.score(baseline, best).total > baseline_score else None


class SelfImprovementLab:
    """Run bounded, auditable self-improvement cycles."""

    def __init__(self, *, benchmark: PolyglotBenchmark | None = None, archive: CandidateArchive | None = None, mutator=None) -> None:
        self.benchmark = benchmark or PolyglotBenchmark()
        self.archive = archive
        self.analyzer = FailureAnalyzer()
        self.mutations = MutationEngine(mutator)
        self.promoter = PromotionEngine()

    def cycle(self, baseline: BenchmarkResult, problem, *, mutation_count: int = 4) -> ImprovementCycle:
        analysis = self.analyzer.analyze(baseline)
        mutations = self.mutations.generate(baseline.candidate, analysis, mutation_count)
        experiments = tuple(self.benchmark.run(item.candidate, problem) for item in mutations)
        decision = self.promoter.decide(baseline, experiments)
        if self.archive is not None:
            for result in experiments:
                self.archive.save(result)
        return ImprovementCycle(baseline, experiments, (analysis,), mutations, decision)

    def iterate(self, baseline: BenchmarkResult, problem, *, rounds: int = 3, mutation_count: int = 4) -> tuple[BenchmarkResult, tuple[ImprovementCycle, ...]]:
        if rounds < 1:
            raise ValueError("rounds must be positive")
        current = baseline
        cycles: list[ImprovementCycle] = []
        for _ in range(rounds):
            cycle = self.cycle(current, problem, mutation_count=mutation_count)
            cycles.append(cycle)
            promoted = self.promoter.select_promoted(current, cycle.experiments)
            if promoted is None:
                break
            current = promoted
        return current, tuple(cycles)
