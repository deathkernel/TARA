"""Evidence-gated orchestration for real TARA model evaluation.

This module coordinates baseline/candidate checkpoint evaluation and promotion
without performing training itself. Training remains an explicit injected
operation so normal runtime code cannot unexpectedly consume large compute.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .capability_suite import capability_cases
from .intelligence_benchmark import BenchmarkReport, RegressionGate
from .model_capability_runner import ModelEvaluation, evaluate_checkpoint


@dataclass(frozen=True)
class EvaluationDecision:
    accepted: bool
    reason: str
    baseline_score: float
    candidate_score: float
    regression: bool
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationCycle:
    baseline: ModelEvaluation
    candidate: ModelEvaluation | None
    decision: EvaluationDecision

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": None if self.candidate is None else self.candidate.as_dict(),
            "decision": self.decision.as_dict(),
        }


class CheckpointEvaluator:
    """Compare checkpoints using the same immutable capability suite."""

    def __init__(self, *, max_new_tokens: int = 32, overall_tolerance: float = 0.0, category_tolerance: float = 0.0):
        self.max_new_tokens = max_new_tokens
        self.gate = RegressionGate(overall_tolerance=overall_tolerance, category_tolerance=category_tolerance)

    def evaluate(self, checkpoint: str | Path, *, name: str = "tara-real-model") -> ModelEvaluation:
        return evaluate_checkpoint(checkpoint, capability_cases(), name=name, max_new_tokens=self.max_new_tokens)

    def compare(self, baseline: ModelEvaluation, candidate: ModelEvaluation) -> EvaluationDecision:
        regression = self.gate.check(baseline.benchmark, candidate.benchmark)
        base = baseline.benchmark.overall_score
        current = candidate.benchmark.overall_score
        if regression:
            reason = "candidate failed the regression gate"
            accepted = False
        elif current <= base:
            reason = "candidate did not strictly improve the overall benchmark"
            accepted = False
        else:
            reason = "candidate improved overall score without regression"
            accepted = True
        fingerprint = sha256(json.dumps({"baseline": baseline.fingerprint, "candidate": candidate.fingerprint, "accepted": accepted}, sort_keys=True).encode()).hexdigest()
        return EvaluationDecision(accepted, reason, base, current, regression, fingerprint)


class EvaluationOrchestrator:
    """Run baseline evaluation, optional injected improvement, and promotion."""

    def __init__(self, evaluator: CheckpointEvaluator | None = None):
        self.evaluator = evaluator or CheckpointEvaluator()

    def run(self, baseline_checkpoint: str | Path, *, improve: Callable[[ModelEvaluation], str | Path] | None = None) -> EvaluationCycle:
        baseline = self.evaluator.evaluate(baseline_checkpoint, name="tara-baseline")
        if improve is None:
            decision = EvaluationDecision(False, "no improvement experiment was supplied", baseline.benchmark.overall_score, baseline.benchmark.overall_score, False, baseline.fingerprint)
            return EvaluationCycle(baseline, None, decision)
        candidate_path = improve(baseline)
        candidate = self.evaluator.evaluate(candidate_path, name="tara-candidate")
        return EvaluationCycle(baseline, candidate, self.evaluator.compare(baseline, candidate))


def write_cycle(cycle: EvaluationCycle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cycle.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination


__all__ = ["CheckpointEvaluator", "EvaluationCycle", "EvaluationDecision", "EvaluationOrchestrator", "write_cycle"]
