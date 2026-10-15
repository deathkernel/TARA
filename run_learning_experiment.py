"""Run TARA's explicit train -> evaluate -> gate learning cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning_experiment import LearningExperiment
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig
from src.teacher_training import validate_approved_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an auditable TARA learning experiment from human-approved Ollama teacher data")
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("candidate_checkpoint")
    parser.add_argument("--report", default=None)
    parser.add_argument("--allow-legacy-data", action="store_true", help="explicitly bypass teacher approval validation for a legacy experiment")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.allow_legacy_data:
        validate_approved_dataset(args.dataset)

    baseline = Path(args.baseline_checkpoint).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    candidate = Path(args.candidate_checkpoint).expanduser().resolve()
    training_manifest = Path(args.manifest).expanduser().resolve() if args.manifest else candidate.with_suffix(".training.json")
    destinations = [("candidate checkpoint", candidate), ("training manifest", training_manifest)]
    if args.metrics:
        destinations.append(("metrics", Path(args.metrics).expanduser().resolve()))
    if args.report:
        destinations.append(("evidence report", Path(args.report).expanduser().resolve()))
    resolved = [path.resolve() for _, path in destinations]
    if len({baseline, dataset, *resolved}) != 2 + len(resolved):
        parser.error("baseline, dataset, candidate, metrics, manifest, and report paths must be distinct")
    for label, path in destinations:
        if path.exists():
            parser.error(f"refusing to overwrite existing {label}: {path}")

    config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, context=args.context, lr=args.lr)
    trainer = TrainingExperiment(config=config, device=args.device)
    def train_candidate(_baseline):
        report = trainer.run(
            args.dataset,
            args.candidate_checkpoint,
            metrics_path=args.metrics,
            manifest_path=training_manifest,
        )
        return report.checkpoint

    report = LearningExperiment().run(
        args.baseline_checkpoint,
        train_candidate=train_candidate,
        output=args.report,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True, default=str))
    raise SystemExit(0 if report.decision.accepted else 2)


if __name__ == "__main__":
    main()
