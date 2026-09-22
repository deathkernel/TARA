"""Evidence-gated orchestration for real TARA model evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .capability_suite import capability_cases
from .intelligence_benchmark import compare_regression
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
    """Compare checkpoints using the same deterministic capability suite."""

    benchmark_name = "tara-capability-v1"

    def __init__(self, *, max_new_tokens: int = 32, overall_tolerance: float = 0.0, category_tolerance: float = 0.0):
        self.max_new_tokens = max_new_tokens
        self.overall_tolerance = overall_tolerance
        self.category_tolerance = category_tolerance

    def evaluate(self, checkpoint: str | Path, *, name: str | None = None) -> ModelEvaluation:
        return evaluate_checkpoint(checkpoint, capability_cases(), name=name or self.benchmark_name, max_new_tokens=self.max_new_tokens)

    def compare(self, baseline: ModelEvaluation, candidate: ModelEvaluation) -> EvaluationDecision:
        gate = compare_regression(
            baseline.benchmark,
            candidate.benchmark,
            minimum_overall=baseline.benchmark.overall_score - self.overall_tolerance,
            maximum_category_drop=self.category_tolerance,
        )
        base = baseline.benchmark.overall_score
        current = candidate.benchmark.overall_score
        if not gate.passed:
            reason = "candidate failed the regression gate"
            accepted = False
        elif current <= base:
            reason = "candidate did not strictly improve the overall benchmark"
            accepted = False
        else:
            reason = "candidate improved overall score without regression"
            accepted = True
        fingerprint = sha256(json.dumps({"baseline": baseline.fingerprint, "candidate": candidate.fingerprint, "accepted": accepted}, sort_keys=True).encode("utf-8")).hexdigest()
        return EvaluationDecision(accepted, reason, base, current, bool(gate.regressions), fingerprint)


class EvaluationOrchestrator:
    """Evaluate a baseline, optionally evaluate an injected candidate, then decide."""

    def __init__(self, evaluator: CheckpointEvaluator | None = None):
        self.evaluator = evaluator or CheckpointEvaluator()

    def run(self, baseline_checkpoint: str | Path, *, improve: Callable[[ModelEvaluation], str | Path] | None = None) -> EvaluationCycle:
        baseline = self.evaluator.evaluate(baseline_checkpoint)
        if improve is None:
            decision = EvaluationDecision(False, "no improvement experiment was supplied", baseline.benchmark.overall_score, baseline.benchmark.overall_score, False, baseline.fingerprint)
            return EvaluationCycle(baseline, None, decision)
        candidate = self.evaluator.evaluate(improve(baseline))
        return EvaluationCycle(baseline, candidate, self.evaluator.compare(baseline, candidate))


def write_cycle(cycle: EvaluationCycle, path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(cycle.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return destination


__all__ = ["CheckpointEvaluator", "EvaluationCycle", "EvaluationDecision", "EvaluationOrchestrator", "write_cycle"]
