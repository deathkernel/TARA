"""Post-training evidence consolidation for TARA.

Phase 37.12 turns an executed learning result into durable cognitive memory.
The module never promotes a checkpoint by itself: callers provide the measured
result and an explicit promotion decision. This keeps training/evaluation
separate from persistent memory mutation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cognitive_memory import CognitiveMemory


@dataclass(frozen=True)
class ConsolidationReport:
    cycle_id: str
    checkpoint: str
    dataset_fingerprint: str
    outcome: str
    promoted: bool
    validation_loss: float | None
    regression_passed: bool | None
    memory_ids: tuple[str, ...]
    fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class LearningConsolidator:
    """Persist measured learning outcomes and their provenance."""

    def __init__(self, memory: CognitiveMemory) -> None:
        self.memory = memory

    @staticmethod
    def _cycle_id(checkpoint: str | Path, dataset_fingerprint: str, outcome: str) -> str:
        payload = f"{checkpoint}|{dataset_fingerprint}|{outcome}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def consolidate(
        self,
        *,
        checkpoint: str | Path,
        dataset_fingerprint: str,
        validation_loss: float | None,
        regression_passed: bool | None,
        promoted: bool,
        outcome: str,
        metrics: dict[str, Any] | None = None,
    ) -> ConsolidationReport:
        if not outcome.strip():
            raise ValueError("outcome must be non-empty")
        cycle_id = self._cycle_id(checkpoint, dataset_fingerprint, outcome)
        payload = {
            "cycle_id": cycle_id,
            "checkpoint": str(checkpoint),
            "dataset_fingerprint": dataset_fingerprint,
            "outcome": outcome,
            "promoted": bool(promoted),
            "validation_loss": validation_loss,
            "regression_passed": regression_passed,
            "metrics": metrics or {},
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        ids: list[str] = []
        ids.append(self.memory.remember(
            kind="learning_cycle",
            domain="continual_learning",
            content=payload,
            confidence=1.0 if regression_passed is True else 0.5,
            verification="verified" if regression_passed is True else "unverified",
            provenance={"component": "learning_consolidation", "cycle_id": cycle_id},
            attributes={"promoted": bool(promoted), "fingerprint": fingerprint},
        ))
        if promoted:
            ids.append(self.memory.remember(
                kind="model_checkpoint",
                domain="continual_learning",
                content={"checkpoint": str(checkpoint), "dataset_fingerprint": dataset_fingerprint, "cycle_id": cycle_id},
                confidence=1.0 if regression_passed is True else 0.5,
                verification="verified" if regression_passed is True else "unverified",
                provenance={"learning_cycle_id": cycle_id},
                attributes={"promoted": True},
            ))
            self.memory.link(ids[0], "produced_checkpoint", ids[1])
        return ConsolidationReport(
            cycle_id=cycle_id,
            checkpoint=str(checkpoint),
            dataset_fingerprint=dataset_fingerprint,
            outcome=outcome,
            promoted=bool(promoted),
            validation_loss=validation_loss,
            regression_passed=regression_passed,
            memory_ids=tuple(ids),
            fingerprint=fingerprint,
        )


__all__ = ["ConsolidationReport", "LearningConsolidator"]
