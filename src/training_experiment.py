"""Auditable training orchestration for TARA Phase 37.5.

The runner makes training an explicit experiment: audit the corpus, train with
an explicit configuration, and persist a reproducible manifest. It never
silently starts training from normal runtime code.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.dataset_audit import DatasetAuditor
from src.training_pipeline import TrainingConfig, TrainingPipeline, TrainingSummary, load_training_texts


@dataclass(frozen=True)
class TrainingExperimentReport:
    experiment_id: str
    dataset: str
    dataset_fingerprint: str
    audit_fingerprint: str
    records: int
    checkpoint: str
    metrics_path: str | None
    start_step: int
    final_step: int
    train_loss: float
    validation_loss: float | None
    stopped_early: bool
    device: str
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class TrainingExperiment:
    """Run a validated, reproducible offline training experiment."""

    def __init__(self, config: TrainingConfig | None = None, device: str | None = None, *, min_length: int = 8) -> None:
        self.config = config or TrainingConfig()
        self.device = device
        self.auditor = DatasetAuditor(min_length=min_length)

    @staticmethod
    def _experiment_id(dataset_fingerprint: str, config: TrainingConfig) -> str:
        payload = json.dumps({"dataset": dataset_fingerprint, "config": asdict(config)}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def audit(self, data: str | Path):
        return self.auditor.audit_jsonl(data)

    def run(
        self,
        data: str | Path,
        output: str | Path,
        *,
        resume: str | Path | None = None,
        metrics_path: str | Path | None = None,
        manifest_path: str | Path | None = None,
        allow_warnings: bool = True,
    ) -> TrainingExperimentReport:
        records = load_training_texts(data)
        audit = self.auditor.audit(records)
        if not audit.healthy:
            raise ValueError("dataset audit contains error-level issues; training blocked")
        if not allow_warnings and audit.issues:
            raise ValueError("dataset audit contains warnings; set allow_warnings=True to continue")

        summary: TrainingSummary = TrainingPipeline(self.config, device=self.device).train(
            data=data, output=output, resume=resume, metrics_path=metrics_path
        )
        experiment_id = self._experiment_id(summary.dataset_fingerprint, self.config)
        report = TrainingExperimentReport(
            experiment_id=experiment_id,
            dataset=str(data),
            dataset_fingerprint=summary.dataset_fingerprint,
            audit_fingerprint=audit.fingerprint,
            records=audit.records,
            checkpoint=summary.checkpoint,
            metrics_path=summary.metrics_path,
            start_step=summary.start_step,
            final_step=summary.final_step,
            train_loss=summary.train_loss,
            validation_loss=summary.validation_loss,
            stopped_early=summary.stopped_early,
            device=summary.device,
            config=asdict(self.config),
        )
        if manifest_path is not None:
            path = Path(manifest_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
