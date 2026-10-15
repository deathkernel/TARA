"""Train TARA's experimental research decoder from scratch.

Example:
    python train_research_lm.py --dataset tinystories --steps 500 --device cpu

The existing train_fast.py and FastTinyLanguageModel remain unchanged as the
baseline. Compare results using identical tokenizer/data/seed/budget settings.
"""
import argparse
import random
import time

import torch

from src.research_language_model import ResearchTinyLanguageModel, count_parameters
from src.text_dataset import list_datasets, load_dataset_text
from src.tokenizer import BPETokenizer, CharTokenizer


def make_windows(ids, context_length):
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if len(ids) <= context_length:
        raise ValueError("dataset is too short for the chosen context length")
    return [
        (ids[start : start + context_length], ids[start + 1 : start + context_length + 1])
        for start in range(len(ids) - context_length)
    ]


def sample_batch(windows, batch_size, rng, device):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    selected = [rng.choice(windows) for _ in range(batch_size)]
    inputs = torch.tensor([item[0] for item in selected], dtype=torch.long, device=device)
    targets = torch.tensor([item[1] for item in selected], dtype=torch.long, device=device)
    return inputs, targets


def evaluate(model, windows, batch_size, batches, rng, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_batch(windows, batch_size, rng, device)
            total_loss += model.loss(inputs, targets).item()
    return total_loss / batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list_datasets(), default="tinystories")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--train-chars", type=int, default=16384)
    parser.add_argument("--validation-chars", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.train_chars <= 0 or args.validation_chars <= 0:
        parser.error("train and validation character budgets must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device)

    train_text = load_dataset_text(args.dataset, args.train_chars, "train")
    validation_text = load_dataset_text(args.dataset, args.validation_chars, "validation")
    if args.tokenizer == "char":
        tokenizer = CharTokenizer(train_text)
    else:
        tokenizer = BPETokenizer(train_text, vocab_size=args.vocab_size)

    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)
    train_windows = make_windows(train_ids, args.context_length)
    validation_windows = make_windows(validation_ids, args.context_length)

    model = ResearchTinyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=args.embedding_dim,
        ff_dim=args.ff_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        max_context=args.context_length,
        num_layers=args.num_layers,
        dropout=args.dropout,
        seed=args.seed,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    started = time.perf_counter()
    for step in range(args.steps):
        inputs, targets = sample_batch(train_windows, args.batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs, targets)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == args.steps - 1:
            print(f"step={step:4d} train_loss={loss.item():.6f}")

    elapsed = time.perf_counter() - started
    validation_batches = min(16, max(1, len(validation_windows) // args.batch_size))
    validation_loss = evaluate(
        model, validation_windows, args.batch_size, validation_batches, rng, device
    )
    print()
    print("Architecture: ResearchTinyLanguageModel (RoPE + GQA + RMSNorm + SwiGLU)")
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(validation_ids)}")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Validation loss: {validation_loss:.6f}")
    print(f"Training time: {elapsed:.3f}s")
    print(f"Seconds/step: {elapsed / args.steps:.4f}")
    print(f"Tokens/step: {args.batch_size * args.context_length}")


if __name__ == "__main__":
    main()
