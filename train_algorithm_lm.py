"""CLI for training TARA's algorithm-language model."""

import argparse
from pathlib import Path

from src.continual_learning import ReplayCorpusBuilder
from src.training_pipeline import TrainingConfig, TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TARA's algorithm language model")
    parser.add_argument("--data", default="data/algorithm_tasks.jsonl")
    parser.add_argument("--output", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--resume", default=None, help="resume from a compatible checkpoint")
    parser.add_argument("--metrics-path", default=None, help="append-only JSONL experiment metrics path")
    parser.add_argument("--replay-data", default=None, help="verified knowledge JSONL for continual-learning replay")
    parser.add_argument("--replay-ratio", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
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
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda")
    args = parser.parse_args()

    data_path = Path(args.data)
    if args.replay_data:
        replay_path = data_path.with_name(data_path.stem + ".replay.jsonl")
        report = ReplayCorpusBuilder(replay_ratio=args.replay_ratio, seed=args.seed).merge(data_path, args.replay_data, replay_path)
        print(f"replay eligible={report.eligible_records} selected={report.selected_records} duplicates_removed={report.duplicates_removed} output={report.output_path}")
        data_path = replay_path

    config = TrainingConfig(
        steps=args.steps,
        batch_size=args.batch_size,
        context=args.context,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        heads=args.heads,
        lr=args.lr,
        validation_split=args.validation_split,
        seed=args.seed,
        log_every=args.log_every,
        checkpoint_every=args.checkpoint_every,
        grad_clip=args.grad_clip,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )
    device = None if args.device in (None, "auto") else args.device
    summary = TrainingPipeline(config, device=device).train(data=data_path, output=args.output, resume=args.resume, metrics_path=args.metrics_path)
    print(f"saved={summary.checkpoint} final_step={summary.final_step} train_loss={summary.train_loss:.4f} val_loss={summary.validation_loss if summary.validation_loss is not None else 'n/a'} stopped_early={summary.stopped_early} metrics={summary.metrics_path}")


if __name__ == "__main__":
    main()
