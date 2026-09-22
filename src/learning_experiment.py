"""End-to-end offline learning experiment orchestration.

Phase 37.10 connects real checkpoint evaluation to an explicit training
experiment. Training is injected rather than silently executed, and a
candidate is accepted only when measured improvement survives regression
checks. No checkpoint is overwritten by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable

from .evaluation_orchestrator import CheckpointEvaluator, EvaluationDecision
from .model_capability_runner import ModelEvaluation


@dataclass(frozen=True)
class CapabilityDelta:
    category: str
    baseline: float
    candidate: float
    delta: float


@dataclass(frozen=True)
class LearningExperimentReport:
    baseline: ModelEvaluation
    candidate: ModelEvaluation | None
    decision: EvaluationDecision
    capability_deltas: tuple[CapabilityDelta, ...]
    experiment_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.as_dict(),
            "candidate": None if self.candidate is None else self.candidate.as_dict(),
            "decision": self.decision.as_dict(),
            "capability_deltas": [asdict(item) for item in self.capability_deltas],
            "experiment_fingerprint": self.experiment_fingerprint,
        }


class LearningExperiment:
    """Run an explicit train/evaluate/promotion experiment."""

    def __init__(self, evaluator: CheckpointEvaluator | None = None):
        self.evaluator = evaluator or CheckpointEvaluator()

    @staticmethod
    def _deltas(baseline: ModelEvaluation, candidate: ModelEvaluation) -> tuple[CapabilityDelta, ...]:
        base = {item.category: item.score for item in baseline.benchmark.categories}
        current = {item.category: item.score for item in candidate.benchmark.categories}
        return tuple(
            CapabilityDelta(name, base[name], current[name], current[name] - base[name])
            for name in sorted(base.keys() & current.keys())
        )

    def run(
        self,
        baseline_checkpoint: str | Path,
        *,
        train_candidate: Callable[[ModelEvaluation], str | Path],
        output: str | Path | None = None,
    ) -> LearningExperimentReport:
        baseline = self.evaluator.evaluate(baseline_checkpoint)
        candidate_path = train_candidate(baseline)
        if not Path(candidate_path).is_file():
            raise FileNotFoundError(f"training runner did not produce a checkpoint: {candidate_path}")
        candidate = self.evaluator.evaluate(candidate_path)
        decision = self.evaluator.compare(baseline, candidate)
        report = LearningExperimentReport(
            baseline=baseline,
            candidate=candidate,
            decision=decision,
            capability_deltas=self._deltas(baseline, candidate),
            experiment_fingerprint=sha256(json.dumps({
                "baseline": baseline.fingerprint,
                "candidate": candidate.fingerprint,
                "decision": decision.fingerprint,
            }, sort_keys=True).encode("utf-8")).hexdigest(),
        )
        if output is not None:
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        return report


__all__ = ["CapabilityDelta", "LearningExperiment", "LearningExperimentReport"]
