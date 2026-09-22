"""Run TARA's explicit train -> evaluate -> gate learning cycle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.learning_experiment import LearningExperiment
from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an auditable TARA learning experiment")
    parser.add_argument("baseline_checkpoint")
    parser.add_argument("dataset")
    parser.add_argument("candidate_checkpoint")
    parser.add_argument("--report", default=None)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = TrainingConfig(steps=args.steps, batch_size=args.batch_size, context=args.context, lr=args.lr)
    trainer = TrainingExperiment(config=config, device=args.device)
    training_manifest = Path(args.manifest) if args.manifest else Path(args.candidate_checkpoint).with_suffix(".training.json")

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
