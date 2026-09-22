"""CLI for training TARA's algorithm-language model.

The implementation delegates dataset loading, validation, checkpointing and
resume semantics to ``src.training_pipeline``. Training is still an explicit
offline operation; running this script is what performs the actual training.
"""

import argparse

from src.training_pipeline import TrainingConfig, TrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TARA's algorithm language model")
    parser.add_argument("--data", default="data/algorithm_tasks.jsonl")
    parser.add_argument("--output", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--resume", default=None, help="resume from a Phase 14 checkpoint")
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
    parser.add_argument("--device", default=None, help="auto, cpu, or cuda")
    args = parser.parse_args()

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
    )
    device = None if args.device in (None, "auto") else args.device
    summary = TrainingPipeline(config, device=device).train(
        data=args.data,
        output=args.output,
        resume=args.resume,
    )
    print(
        f"saved={summary.checkpoint} final_step={summary.final_step} "
        f"train_loss={summary.train_loss:.4f} "
        f"val_loss={summary.validation_loss if summary.validation_loss is not None else 'n/a'}"
    )


if __name__ == "__main__":
    main()
