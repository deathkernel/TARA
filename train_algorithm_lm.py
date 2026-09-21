"""Train TARA's small PyTorch LM on algorithm/problem-solving examples.

This is the first algorithm-reasoning training stage. It teaches the LM to
continue structured problem -> approach text. Verification remains a separate
runtime loop in ``src.algorithm_discovery.py``.
"""

import argparse
import json
import random
from pathlib import Path

import torch

from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def load_texts(path: Path) -> list[str]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            records.append(
                "Problem: " + item["problem"]
                + "\nApproach: " + item["solution"]
                + "\nTests: " + item.get("tests", "")
            )
    if not records:
        raise ValueError("algorithm dataset is empty")
    return records


def make_examples(texts, tokenizer, context):
    ids = []
    for text in texts:
        encoded = tokenizer.encode(text)
        if len(encoded) >= 2:
            ids.extend(encoded + [tokenizer.stoi.get("\n", 0)])
    if len(ids) <= context:
        raise ValueError("dataset is shorter than the requested context")
    return torch.tensor(ids, dtype=torch.long)


def sample_batch(tokens, batch_size, context, device):
    starts = torch.randint(0, len(tokens) - context - 1, (batch_size,))
    x = torch.stack([tokens[i : i + context] for i in starts])
    y = torch.stack([tokens[i + 1 : i + context + 1] for i in starts])
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/algorithm_tasks.jsonl")
    parser.add_argument("--output", default="checkpoints/algorithm_lm.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("steps and batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = load_texts(Path(args.data))
    tokenizer = CharTokenizer("\n".join(texts))
    tokens = make_examples(texts, tokenizer, args.context)

    model = FastTinyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.heads,
        max_context=args.context,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(tokens, args.batch_size, args.context, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(f"step={step:5d} loss={loss.item():.4f} device={device}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "embedding_dim": args.embedding_dim,
                "ff_dim": args.ff_dim,
                "num_heads": args.heads,
                "max_context": args.context,
            },
            "tokenizer": {
                "itos": tokenizer.itos,
                "stoi": tokenizer.stoi,
            },
            "seed": args.seed,
        },
        output,
    )
    print(f"saved={output}")


if __name__ == "__main__":
    main()
