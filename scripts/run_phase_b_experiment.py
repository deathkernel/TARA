"""Run a curated-dataset Phase B experiment against a real baseline checkpoint.

The candidate is written to a distinct, unused path. Evaluation is evidence-only:
this command never promotes, replaces, or deletes any checkpoint.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.evaluation_orchestrator import CheckpointEvaluator
from src.learning_experiment import LearningExperiment
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def _resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def run_phase_b_experiment(
    baseline_checkpoint: str | Path,
    dataset: str | Path,
    candidate_checkpoint: str | Path,
    *,
    steps: int = 100,
    batch_size: int = 16,
    context: int = 64,
    embedding_dim: int = 64,
    ff_dim: int = 128,
    heads: int = 4,
    num_layers: int = 2,
    lr: float = 3e-4,
    validation_split: float = 0.1,
    seed: int = 42,
    max_new_tokens: int = 32,
    device: str | None = None,
    metrics_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
    evaluator: CheckpointEvaluator | None = None,
) -> dict[str, Any]:
    """Train one candidate from reviewed data, benchmark it, and persist the decision."""
    baseline = _resolve(baseline_checkpoint)
    data = _resolve(dataset)
    candidate = _resolve(candidate_checkpoint)
    metrics = _resolve(metrics_path or candidate.with_suffix(".metrics.jsonl"))
    manifest = _resolve(manifest_path or candidate.with_suffix(".training.json"))
    report_file = _resolve(report_path or candidate.with_suffix(".phase-b.json"))
    evaluation_file = candidate.with_suffix(".evaluation.json").resolve()

    if not baseline.is_file():
        raise FileNotFoundError(f"baseline checkpoint does not exist: {baseline}")
    if not data.is_file():
        raise FileNotFoundError(f"reviewed training dataset does not exist: {data}")
    if candidate == baseline:
        raise ValueError("candidate checkpoint path must differ from baseline checkpoint path")

    input_paths = {baseline, data}
    output_paths = [candidate, metrics, manifest, report_file, evaluation_file]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("candidate, metrics, manifest, evaluation, and report paths must be distinct")
    if input_paths.intersection(output_paths):
        raise ValueError("output paths must not overlap the baseline checkpoint or dataset")
    existing = [str(path) for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing Phase B outputs: " + ", ".join(existing))

    config = TrainingConfig(
        steps=steps,
        batch_size=batch_size,
        context=context,
        embedding_dim=embedding_dim,
        ff_dim=ff_dim,
        heads=heads,
        num_layers=num_layers,
        lr=lr,
        validation_split=validation_split,
        seed=seed,
    )
    training_result: dict[str, Any] = {}

    def train_candidate(_baseline_evaluation):
        trained = TrainingExperiment(config=config, device=device).run(
            data,
            candidate,
            metrics_path=metrics,
            manifest_path=manifest,
        )
        training_result.update(trained.as_dict())
        return candidate

    learning = LearningExperiment(
        evaluator=evaluator or CheckpointEvaluator(max_new_tokens=max_new_tokens)
    ).run(
        baseline,
        train_candidate=train_candidate,
        output=evaluation_file,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "phase-b-reviewed-dataset-experiment",
        "interpretation": (
            "A benchmark decision is evidence for this checkpoint and suite only; "
            "it does not prove general intelligence. No checkpoint is promoted automatically."
        ),
        "baseline_checkpoint": str(baseline),
        "dataset": str(data),
        "candidate_checkpoint": str(candidate),
        "dataset_fingerprint": training_result.get("dataset_fingerprint"),
        "training": training_result,
        "learning_experiment": learning.as_dict(),
        "outputs": {
            "metrics": str(metrics),
            "training_manifest": str(manifest),
            "evaluation": str(evaluation_file),
            "report": str(report_file),
        },
        "promotion_accepted": learning.decision.accepted,
    }
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a candidate on a reviewed dataset, evaluate it, and apply the regression gate"
    )
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("reviewed_dataset", help="curated JSONL/TXT dataset; teacher output must be reviewed first")
    parser.add_argument("candidate_checkpoint", help="new output path; existing files are never overwritten")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to available CUDA/CPU")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = run_phase_b_experiment(
        args.baseline_checkpoint,
        args.reviewed_dataset,
        args.candidate_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        context=args.context,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        heads=args.heads,
        num_layers=args.layers,
        lr=args.lr,
        validation_split=args.validation_split,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        metrics_path=args.metrics,
        manifest_path=args.manifest,
        report_path=args.report,
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not report["promotion_accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
