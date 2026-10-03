"""Train the accelerated PyTorch TARA model on a registered dataset.

This backend keeps the architecture explicit but replaces scalar Python graph
construction with batched tensor operations and PyTorch autograd.

Example:
    python train_fast.py --dataset tinystories --steps 500 --batch-size 16
"""

import argparse
import random
import time

import torch

from src.text_dataset import list_datasets, load_dataset_text
from src.tokenizer import BPETokenizer, CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def make_windows(ids, context_length):
    windows = []
    for start in range(0, len(ids) - context_length):
        windows.append((
            ids[start:start + context_length],
            ids[start + 1:start + context_length + 1],
        ))
    if not windows:
        raise ValueError("dataset is too short for the chosen context length")
    return windows


def sample_batch(windows, batch_size, rng, device):
    selected = [rng.choice(windows) for _ in range(batch_size)]
    inputs = torch.tensor(
        [item[0] for item in selected],
        dtype=torch.long,
        device=device,
    )
    targets = torch.tensor(
        [item[1] for item in selected],
        dtype=torch.long,
        device=device,
    )
    return inputs, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list_datasets(), default="tinystories")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="bpe")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--train-chars", type=int, default=16384)
    parser.add_argument("--validation-chars", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--ff-dim", type=int, default=1024)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    train_text = load_dataset_text(args.dataset, args.train_chars, "train")
    validation_text = load_dataset_text(
        args.dataset,
        args.validation_chars,
        "validation",
    )

    if args.tokenizer == "char":
        tokenizer = CharTokenizer(train_text)
    else:
        tokenizer = BPETokenizer(train_text, vocab_size=args.vocab_size)

    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)

    train_windows = make_windows(train_ids, args.context_length)
    validation_windows = make_windows(validation_ids, args.context_length)

    model = FastTinyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        max_context=args.context_length,
        num_layers=6,
        seed=args.seed,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = random.Random(args.seed)

    model.train()
    started = time.perf_counter()

    for step in range(args.steps):
        inputs, targets = sample_batch(
            train_windows,
            args.batch_size,
            rng,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs, targets)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == args.steps - 1:
            print(f"step={step:4d} loss={loss.item():.6f}")

    elapsed = time.perf_counter() - started

    model.eval()
    with torch.no_grad():
        validation_loss = 0.0
        batches = min(16, max(1, len(validation_windows) // args.batch_size))
        for _ in range(batches):
            inputs, targets = sample_batch(
                validation_windows,
                args.batch_size,
                rng,
                device,
            )
            validation_loss += model.loss(inputs, targets).item()
        validation_loss /= batches

    print()
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(validation_ids)}")
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Training time: {elapsed:.3f}s")
    print(f"Seconds/step: {elapsed / args.steps:.4f}")
    print(f"Tokens/step: {args.batch_size * args.context_length}")


if __name__ == "__main__":
    main()
