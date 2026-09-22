"""Run an auditable TARA training experiment from the command line."""
from __future__ import annotations

import argparse
import json

from src.training_experiment import TrainingExperiment
from src.training_pipeline import TrainingConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible TARA training experiment")
    parser.add_argument("--data", default="data/algorithm_tasks.jsonl")
    parser.add_argument("--output", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no-tie-embeddings", action="store_true")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = TrainingConfig(
        steps=args.steps, batch_size=args.batch_size, context=args.context,
        embedding_dim=args.embedding_dim, ff_dim=args.ff_dim, heads=args.heads,
        num_layers=args.num_layers, dropout=args.dropout, tie_embeddings=not args.no_tie_embeddings,
        lr=args.lr, validation_split=args.validation_split, seed=args.seed,
        log_every=args.log_every, checkpoint_every=args.checkpoint_every,
        grad_clip=args.grad_clip, gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps, min_lr_ratio=args.min_lr_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    report = TrainingExperiment(config, device=args.device).run(
        args.data, args.output, resume=args.resume,
        metrics_path=args.metrics_path, manifest_path=args.manifest_path,
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
