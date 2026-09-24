"""Run a bounded TARA vNext conversational training experiment."""

import argparse
import random
import time
from pathlib import Path

import torch

from src.dataset_registry import get_dataset_spec
from src.text_dataset import load_dataset_text, list_datasets
from src.tokenizer import BPETokenizer
from src.torch_language_model import FastTinyLanguageModel


def windows(ids, context):
    if len(ids) <= context:
        raise ValueError("tokenized dataset is too short for context")
    return [(ids[i:i+context], ids[i+1:i+context+1]) for i in range(len(ids)-context)]


def batch(data, size, rng, device):
    chosen = [data[rng.randrange(len(data))] for _ in range(size)]
    return (
        torch.tensor([x for x, _ in chosen], dtype=torch.long, device=device),
        torch.tensor([y for _, y in chosen], dtype=torch.long, device=device),
    )


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list_datasets(), default="soda")
    p.add_argument("--train-chars", type=int, default=262144)
    p.add_argument("--validation-chars", type=int, default=65536)
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--embedding", type=int, default=256)
    p.add_argument("--ff-dim", type=int, default=1024)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--checkpoint", default="checkpoints/tara_vnext_soda.pt")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    spec = get_dataset_spec(args.dataset)
    validation_split = spec.validation_split
    if validation_split is None:
        raise ValueError(f"{spec.name} does not define a validation split")

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    t0 = time.perf_counter()
    print(f"[1/5] Loading {spec.name} train split...", flush=True)
    train_text = load_dataset_text(
        args.dataset, args.train_chars, spec.train_split, seed=args.seed
    )
    print(f"[1/5] Train text: {len(train_text):,} chars", flush=True)

    print(f"[2/5] Loading {spec.name} validation split...", flush=True)
    val_text = load_dataset_text(
        args.dataset, args.validation_chars, validation_split, seed=args.seed + 1
    )
    print(f"[2/5] Validation text: {len(val_text):,} chars", flush=True)

    print("[3/5] Training BPE tokenizer...", flush=True)
    tokenizer = BPETokenizer(train_text, vocab_size=args.vocab_size)
    train = windows(tokenizer.encode(train_text), args.context)
    val = windows(tokenizer.encode(val_text), args.context)
    print(
        f"[3/5] Tokenizer vocab={tokenizer.vocab_size:,} "
        f"train_tokens={len(train)+args.context-0:,} val_tokens={len(val)+args.context-0:,}",
        flush=True,
    )

    model = FastTinyLanguageModel(
        tokenizer.vocab_size,
        args.embedding,
        args.ff_dim,
        args.heads,
        args.context,
        args.layers,
        0.1,
        True,
        args.seed,
    ).to(device)
    print(
        f"[4/5] Model params={parameter_count(model):,} device={device}",
        flush=True,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    best = float("inf")
    train_start = time.perf_counter()

    print("[5/5] Training...", flush=True)
    for step in range(1, args.steps + 1):
        step_start = time.perf_counter()
        model.train()
        x, y = batch(train, args.batch_size, rng, device)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        elapsed = time.perf_counter() - step_start
        tokens_per_step = args.batch_size * args.context
        tok_s = tokens_per_step / elapsed if elapsed > 0 else 0.0

        if step == 1 or step % 10 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = batch(val, args.batch_size, rng, device)
                vloss = float(model.loss(vx, vy).item())

            print(
                f"step={step:4d} train_loss={loss.item():.4f} "
                f"val_loss={vloss:.4f} device={device} "
                f"sec/step={elapsed:.2f} tokens/sec={tok_s:,.0f}",
                flush=True,
            )

            if vloss < best:
                best = vloss
                Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "format_version": 6,
                        "model_state": model.state_dict(),
                        "model_config": {
                            "vocab_size": tokenizer.vocab_size,
                            "embedding_dim": args.embedding,
                            "ff_dim": args.ff_dim,
                            "num_heads": args.heads,
                            "max_context": args.context,
                            "num_layers": args.layers,
                            "dropout": 0.1,
                            "tie_embeddings": True,
                        },
                        "tokenizer": {
                            "type": "bpe",
                            "itos": tokenizer.itos,
                            "stoi": tokenizer.stoi,
                            "merges": tokenizer.merges,
                        },
                        "step": step,
                        "validation_loss": vloss,
                        "train_loss": float(loss.item()),
                        "dataset": args.dataset,
                        "seed": args.seed,
                    },
                    args.checkpoint,
                )

    total_train = time.perf_counter() - train_start
    print(
        f"training_seconds={total_train:.2f} checkpoint={args.checkpoint} "
        f"best_validation_loss={best:.4f}",
        flush=True,
    )
    print(f"prepare_seconds={time.perf_counter() - t0 - total_train:.2f}", flush=True)


if __name__ == "__main__":
    main()
