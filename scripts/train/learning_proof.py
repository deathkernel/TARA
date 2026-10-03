"""Controlled proof that TARA's neural core actually learns from examples.

This is deliberately small and self-contained. It does not add a new cognitive
module or curriculum stage. It answers one question only:

    Can the current decoder Transformer reduce next-token loss on training data
    while retaining measurable performance on held-out text?

Run from the repository root:

    python scripts/train/learning_proof.py --device cuda --steps 300

The experiment writes a checkpoint and JSON report locally. Nothing is committed
by the script.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from src.tara_mind.core.tokenizer import FastBPETokenizer
from src.tara_mind.core.transformer import FastTinyLanguageModel


TRAIN_TEXT = """
TARA learns by predicting the next token from context.
A neural language model changes its parameters when prediction errors create gradients.
Learning is not the same thing as adding a rule to the program.
The model should improve because its weights are updated from examples.
TARA can learn simple facts when those facts appear repeatedly in training examples.
The sky is blue.
Water freezes at zero degrees Celsius.
Two plus two equals four.
A triangle has three sides.
A plant needs light to grow.
""".strip()

VALIDATION_TEXT = """
Prediction error provides a signal that changes neural parameters.
A shape with three straight edges is a triangle.
Adding two objects to two objects gives four objects.
Plants commonly use light as an energy source for photosynthesis.
Ice is solid water, while liquid water changes state when heated.
A model can improve its predictions without a hand-written rule for every answer.
""".strip()


def windows(ids: list[int], context: int) -> list[tuple[list[int], list[int]]]:
    if len(ids) <= context:
        raise ValueError("text is too short for the selected context")
    return [
        (ids[i : i + context], ids[i + 1 : i + context + 1])
        for i in range(len(ids) - context)
    ]


def make_batch(
    data: list[tuple[list[int], list[int]]],
    batch_size: int,
    rng: random.Random,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = [data[rng.randrange(len(data))] for _ in range(batch_size)]
    x = torch.tensor([a for a, _ in chosen], dtype=torch.long, device=device)
    y = torch.tensor([b for _, b in chosen], dtype=torch.long, device=device)
    return x, y


@torch.no_grad()
def evaluate(
    model: FastTinyLanguageModel,
    data: list[tuple[list[int], list[int]]],
    batch_size: int,
    device: torch.device,
    seed: int,
    batches: int = 8,
) -> tuple[float, float]:
    model.eval()
    rng = random.Random(seed)
    losses: list[float] = []
    correct = 0
    total = 0
    for _ in range(max(1, batches)):
        x, y = make_batch(data, batch_size, rng, device)
        logits = model(x)
        losses.append(float(torch.nn.functional.cross_entropy(
            logits.reshape(-1, model.vocab_size), y.reshape(-1)
        ).item()))
        correct += int((logits.argmax(dim=-1) == y).sum().item())
        total += y.numel()
    return sum(losses) / len(losses), correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--embedding", type=int, default=128)
    parser.add_argument("--ff-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/learning_proof.pt")
    parser.add_argument("--report", default="outputs/learning_proof.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tokenizer = FastBPETokenizer(TRAIN_TEXT, vocab_size=args.vocab_size)
    train_ids = tokenizer.encode(TRAIN_TEXT)
    validation_ids = tokenizer.encode(VALIDATION_TEXT)

    train_windows = windows(train_ids, args.context)
    validation_windows = windows(validation_ids, args.context)

    model = FastTinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=args.embedding,
        ff_dim=args.ff_dim,
        num_heads=args.heads,
        max_context=args.context,
        num_layers=args.layers,
        dropout=0.0,
        tie_embeddings=True,
        seed=args.seed,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    before_loss, before_accuracy = evaluate(
        model, validation_windows, args.batch_size, device, args.seed + 1
    )

    model.train()
    for step in range(args.steps):
        x, y = make_batch(train_windows, args.batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == args.steps - 1:
            print(f"step={step + 1} train_loss={loss.item():.6f}", flush=True)

    after_loss, after_accuracy = evaluate(
        model, validation_windows, args.batch_size, device, args.seed + 2
    )

    checkpoint = Path(args.output)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "embedding_dim": args.embedding,
                "ff_dim": args.ff_dim,
                "num_heads": args.heads,
                "max_context": args.context,
                "num_layers": args.layers,
                "dropout": 0.0,
                "tie_embeddings": True,
            },
            "tokenizer": {"type": "fast_bpe", "json": tokenizer.to_json()},
            "experiment": "controlled_learning_proof",
        },
        checkpoint,
    )

    report = {
        "experiment": "controlled_learning_proof",
        "device": str(device),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_tokens": len(train_ids),
        "validation_tokens": len(validation_ids),
        "steps": args.steps,
        "before": {"loss": before_loss, "accuracy": before_accuracy},
        "after": {"loss": after_loss, "accuracy": after_accuracy},
        "validation_loss_delta": after_loss - before_loss,
        "validation_accuracy_delta": after_accuracy - before_accuracy,
        "checkpoint": str(checkpoint),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"device={device}")
    print(f"parameters={report['parameters']:,}")
    print(f"validation_loss_before={before_loss:.6f}")
    print(f"validation_loss_after={after_loss:.6f}")
    print(f"validation_accuracy_before={before_accuracy:.4f}")
    print(f"validation_accuracy_after={after_accuracy:.4f}")
    print(f"checkpoint={checkpoint}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
