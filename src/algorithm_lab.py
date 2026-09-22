"""Advanced algorithm discovery orchestration.

This layer turns TARA's existing generation, polyglot execution, verification,
benchmarking, novelty analysis and knowledge extraction primitives into one
research loop. It uses multi-objective scoring and diversity-aware candidate
selection; it does not assume that generated text is novel or correct.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Callable, Iterable, Sequence

from .polyglot.archive import CandidateArchive
from .polyglot.benchmark import BenchmarkResult, PolyglotBenchmark
from .polyglot.candidate import PolyglotCandidate
from .polyglot.knowledge import KnowledgeExtractor, KnowledgeRecord
from .polyglot.novelty import NoveltyAnalyzer, NoveltyReport
from .polyglot.problems import ProblemSpec
from .polyglot.self_improvement import ImprovementResult, SelfImprovementEngine


@dataclass(frozen=True)
class ProblemModel:
    raw: str
    objective: str
    constraints: tuple[str, ...] = ()
    optimization_target: str = "runtime"
    expected_input: str = ""
    expected_output: str = ""


@dataclass(frozen=True)
class CandidateScore:
    correctness: float
    runtime_score: float
    complexity_score: float
    novelty_score: float
    diversity_score: float
    total: float


@dataclass(frozen=True)
class DiscoveryReport:
    problem: ProblemModel
    candidates: tuple[PolyglotCandidate, ...]
    benchmarks: tuple[BenchmarkResult, ...]
    scores: tuple[CandidateScore, ...]
    novelty: tuple[NoveltyReport, ...]
    verified_knowledge: tuple[KnowledgeRecord, ...]


class ProblemInterpreter:
    """Extract a compact machine-readable objective from natural language."""

    def interpret(self, problem: str) -> ProblemModel:
        text = problem.strip()
        if not text:
            raise ValueError("problem must not be empty")
        lowered = text.lower()
        target = "memory" if any(word in lowered for word in ("memory", "space", "ram")) else "runtime"
        constraints = tuple(
            fragment.strip(" .") for fragment in text.replace(";", ".").split(".")
            if any(key in fragment.lower() for key in ("must", "constraint", "limit", "only", "avoid"))
        )
        return ProblemModel(raw=text, objective=text, constraints=constraints, optimization_target=target)


class CandidateDiversity:
    """Cheap token-set similarity used to discourage selecting clones."""

    @staticmethod
    def similarity(left: str, right: str) -> float:
        a, b = set(left.split()), set(right.split())
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def diversity(self, candidate: PolyglotCandidate, selected: Sequence[PolyglotCandidate]) -> float:
        if not selected:
            return 1.0
        return 1.0 - max(self.similarity(candidate.source, other.source) for other in selected)


class MultiObjectiveSelector:
    """Rank verified candidates with correctness, runtime, novelty and diversity signals."""

    def __init__(self, diversity_weight: float = 0.15, novelty_weight: float = 0.10):
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.diversity = CandidateDiversity()

    @staticmethod
    def _novelty_score(result: BenchmarkResult, novelty: Sequence[NoveltyReport]) -> float:
        report = next((item for item in novelty if item.candidate_fingerprint == result.candidate.fingerprint), None)
        if report is None or report.new_to_archive:
            return 1.0
        nearest = max((match.similarity for match in report.nearest), default=1.0)
        return max(0.0, 1.0 - nearest)

    def select(self, results: Sequence[BenchmarkResult], novelty: Sequence[NoveltyReport]) -> tuple[BenchmarkResult, ...]:
        verified = [result for result in results if result.verified]
        if not verified:
            return ()
        runtimes = [max(1e-9, result.total_runtime_ms) for result in verified]
        minimum, maximum = min(runtimes), max(runtimes)
        selected: list[BenchmarkResult] = []
        while verified:
            best = max(verified, key=lambda result: self._utility(result, selected, novelty, minimum, maximum))
            selected.append(best)
            verified.remove(best)
        return tuple(selected)

    def _utility(self, result, selected, novelty, minimum, maximum):
        runtime = 1.0 if maximum == minimum else 1.0 - (result.total_runtime_ms - minimum) / (maximum - minimum)
        novel = self._novelty_score(result, novelty)
        diversity = self.diversity.diversity(result.candidate, [item.candidate for item in selected])
        complexity = 1.0 / sqrt(max(1, len(result.candidate.source)))
        return 0.65 * runtime + 0.10 * complexity + self.novelty_weight * novel + self.diversity_weight * diversity


class AlgorithmDiscoveryLab:
    """Run generate -> compile/execute -> verify -> benchmark -> novelty -> knowledge."""

    def __init__(self, *, benchmark=None, archive: CandidateArchive | None = None, novelty=None):
        self.benchmark = benchmark or PolyglotBenchmark()
        self.archive = archive
        self.novelty = novelty or (NoveltyAnalyzer(archive) if archive is not None else None)
        self.selector = MultiObjectiveSelector()
        self.extractor = KnowledgeExtractor(verified_only=True)
        self.interpreter = ProblemInterpreter()

    def discover(self, problem: ProblemSpec,
                 generator: Callable[[ProblemSpec, int], Iterable[PolyglotCandidate]],
                 *, candidates: int = 8) -> DiscoveryReport:
        if candidates < 1:
            raise ValueError("candidates must be positive")
        model = self.interpreter.interpret(problem.description)
        generated = list(generator(problem, candidates))[:candidates]
        if not generated:
            return DiscoveryReport(model, (), (), (), (), ())
        benchmarks = tuple(self.benchmark.run(candidate, problem) for candidate in generated)
        novelty = tuple(self.novelty.analyze(result.candidate) for result in benchmarks) if self.novelty else ()
        selected = self.selector.select(benchmarks, novelty)
        scores = tuple(self._score(result, novelty) for result in selected)
        knowledge = tuple(self.extractor.extract(selected))
        if self.archive is not None:
            for result in selected:
                self.archive.save(result)
        return DiscoveryReport(model, tuple(generated), tuple(benchmarks), scores, novelty, knowledge)

    @staticmethod
    def _score(result, novelty):
        report = next((item for item in novelty if item.candidate_fingerprint == result.candidate.fingerprint), None)
        novelty_score = 1.0 if report is None or report.new_to_archive else max(0.0, 1.0 - max((m.similarity for m in report.nearest), default=1.0))
        return CandidateScore(
            correctness=1.0 if result.verified else 0.0,
            runtime_score=1.0 / max(1e-9, result.total_runtime_ms),
            complexity_score=1.0 / sqrt(max(1, len(result.candidate.source))),
            novelty_score=novelty_score,
            diversity_score=1.0,
            total=0.65 / max(1e-9, result.total_runtime_ms) + 0.10 / sqrt(max(1, len(result.candidate.source))) + 0.10 * novelty_score + 0.15,
        )


class AdaptiveAlgorithmSearch:
    """Compatibility adapter around TARA's existing iterative improvement engine."""

    def __init__(self, benchmark=None, archive=None):
        self.engine = SelfImprovementEngine(benchmark=benchmark, archive=archive)

    def search(self, problem: str, generator, *, rounds=4, candidates_per_round=8) -> ImprovementResult:
        return self.engine.improve(problem, generator, rounds=rounds, candidates_per_round=candidates_per_round)


def discovery_summary(report: DiscoveryReport) -> dict[str, object]:
    return {
        "problem": report.problem.raw,
        "candidates": len(report.candidates),
        "verified": sum(item.verified for item in report.benchmarks),
        "knowledge_records": len(report.verified_knowledge),
        "novelty_reports": len(report.novelty),
    }
