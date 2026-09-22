"""Evidence-gated capability improvement orchestration for TARA."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

from src.intelligence_benchmark import BenchmarkReport, RegressionGate, compare_regression


@dataclass(frozen=True)
class CapabilityFailure:
    case_id: str
    category: str
    score: float
    error: str | None
    severity: str


@dataclass(frozen=True)
class ImprovementTarget:
    category: str
    cases: tuple[str, ...]
    reason: str
    priority: float


@dataclass(frozen=True)
class ImprovementProposal:
    proposal_id: str
    targets: tuple[ImprovementTarget, ...]
    training_hints: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class ImprovementDecision:
    accepted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    regression: RegressionGate
    proposal: ImprovementProposal


@dataclass(frozen=True)
class IntelligenceImprovementReport:
    baseline: BenchmarkReport
    candidate: BenchmarkReport
    failures: tuple[CapabilityFailure, ...]
    proposal: ImprovementProposal
    decision: ImprovementDecision
    fingerprint: str

    def as_dict(self) -> dict:
        return asdict(self)


class FailureAnalyzer:
    def analyze(self, report: BenchmarkReport) -> tuple[CapabilityFailure, ...]:
        return tuple(
            CapabilityFailure(r.case_id, r.category, r.score, r.error, "critical" if r.score == 0.0 else "warning")
            for r in report.results if r.score < 1.0
        )


class ImprovementPlanner:
    def propose(self, failures: Iterable[CapabilityFailure]) -> ImprovementProposal:
        grouped: dict[str, list[CapabilityFailure]] = {}
        for failure in failures:
            grouped.setdefault(failure.category, []).append(failure)
        targets = []
        for category, items in grouped.items():
            mean_score = sum(item.score for item in items) / len(items)
            targets.append(ImprovementTarget(category, tuple(sorted(i.case_id for i in items)),
                                             f"{len(items)} failing cases; mean score={mean_score:.4f}", 1.0 - mean_score))
        targets.sort(key=lambda item: (-item.priority, item.category))
        hints = tuple(f"focus:{target.category}" for target in targets)
        payload = {"targets": [asdict(t) for t in targets], "hints": hints}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return ImprovementProposal(fingerprint[:16], tuple(targets), hints, fingerprint)


class ImprovementGate:
    def __init__(self, *, minimum_overall: float = 0.0, maximum_category_drop: float = 0.05):
        self.minimum_overall = minimum_overall
        self.maximum_category_drop = maximum_category_drop

    def evaluate(self, baseline: BenchmarkReport, candidate: BenchmarkReport) -> RegressionGate:
        return compare_regression(baseline, candidate, minimum_overall=self.minimum_overall,
                                  maximum_category_drop=self.maximum_category_drop)


CandidateRunner = Callable[[ImprovementProposal], BenchmarkReport]


class IntelligenceImprovementEngine:
    def __init__(self, gate: ImprovementGate | None = None):
        self.analyzer = FailureAnalyzer()
        self.planner = ImprovementPlanner()
        self.gate = gate or ImprovementGate()

    def improve(self, baseline: BenchmarkReport, runner: CandidateRunner) -> IntelligenceImprovementReport:
        failures = self.analyzer.analyze(baseline)
        proposal = self.planner.propose(failures)
        if not failures:
            candidate = baseline
            regression = self.gate.evaluate(baseline, candidate)
            accepted = False
            reason = "baseline has no failing benchmark cases; no improvement experiment required"
        else:
            candidate = runner(proposal)
            regression = self.gate.evaluate(baseline, candidate)
            accepted = regression.passed and regression.overall_delta > 0.0
            reason = "candidate improved without regression" if accepted else "candidate rejected by improvement/regression gate"
        decision = ImprovementDecision(accepted, reason, baseline.overall_score, candidate.overall_score, regression, proposal)
        payload = {"baseline": baseline.fingerprint, "candidate": candidate.fingerprint, "proposal": proposal.fingerprint, "accepted": accepted}
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return IntelligenceImprovementReport(baseline, candidate, failures, proposal, decision, fingerprint)


def write_report(report: IntelligenceImprovementReport, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination
