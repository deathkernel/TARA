"""Compare TARA's baseline and research LM under matched training budgets.

This is a controlled experiment, not an assertion that the research model wins.
Run several seeds and preserve the same data/tokenizer/context/optimizer settings.
"""
import argparse
import random
import time

import torch

from src.research_language_model import ResearchTinyLanguageModel, count_parameters
from src.text_dataset import list_datasets, load_dataset_text
from src.tokenizer import BPETokenizer, CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def make_windows(ids, context_length):
    if context_length <= 0 or len(ids) <= context_length:
        raise ValueError("dataset must contain more tokens than a positive context length")
    return [
        (ids[start : start + context_length], ids[start + 1 : start + context_length + 1])
        for start in range(len(ids) - context_length)
    ]


def sample_batch(windows, batch_size, rng, device):
    selected = [rng.choice(windows) for _ in range(batch_size)]
    inputs = torch.tensor([item[0] for item in selected], dtype=torch.long, device=device)
    targets = torch.tensor([item[1] for item in selected], dtype=torch.long, device=device)
    return inputs, targets


def run_experiment(model, windows, args, device):
    rng = random.Random(args.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    started = time.perf_counter()
    for step in range(args.steps):
        inputs, targets = sample_batch(windows, args.batch_size, rng, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(inputs, targets)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    return elapsed


def evaluate(model, windows, args, device):
    rng = random.Random(args.seed + 1)
    model.eval()
    total = 0.0
    batches = min(16, max(1, len(windows) // args.batch_size))
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = sample_batch(windows, args.batch_size, rng, device)
            total += model.loss(inputs, targets).item()
    return total / batches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list_datasets(), default="tinystories")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--train-chars", type=int, default=16384)
    parser.add_argument("--validation-chars", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--ff-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        parser.error("--steps and --batch-size must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_text = load_dataset_text(args.dataset, args.train_chars, "train")
    validation_text = load_dataset_text(args.dataset, args.validation_chars, "validation")
    tokenizer = (
        CharTokenizer(train_text)
        if args.tokenizer == "char"
        else BPETokenizer(train_text, vocab_size=args.vocab_size)
    )
    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)
    train_windows = make_windows(train_ids, args.context_length)
    validation_windows = make_windows(validation_ids, args.context_length)

    model_specs = (
        (
            "baseline",
            lambda: FastTinyLanguageModel(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=args.embedding_dim,
                ff_dim=args.ff_dim,
                num_heads=args.num_heads,
                max_context=args.context_length,
                num_layers=args.num_layers,
                seed=args.seed,
            ),
        ),
        (
            "research_rope_gqa_swiglu",
            lambda: ResearchTinyLanguageModel(
                vocab_size=tokenizer.vocab_size,
                embedding_dim=args.embedding_dim,
                ff_dim=args.ff_dim,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
                max_context=args.context_length,
                num_layers=args.num_layers,
                seed=args.seed,
            ),
        ),
    )

    print(
        f"dataset={args.dataset} tokenizer={args.tokenizer} "
        f"train_tokens={len(train_ids)} validation_tokens={len(validation_ids)} "
        f"seed={args.seed} steps={args.steps} batch={args.batch_size} "
        f"context={args.context_length}"
    )
    results = []
    for name, make_model in model_specs:
        model = make_model().to(device)
        # Re-seed the batch sampler per architecture for matched batch choices.
        elapsed = run_experiment(model, train_windows, args, device)
        validation_loss = evaluate(model, validation_windows, args, device)
        parameters = count_parameters(model)
        tokens_per_second = (
            args.steps * args.batch_size * args.context_length / elapsed
            if elapsed > 0
            else float("inf")
        )
        results.append((name, parameters, validation_loss, elapsed, tokens_per_second))
        print(
            f"model={name} parameters={parameters} validation_loss={validation_loss:.6f} "
            f"train_seconds={elapsed:.3f} train_tokens_per_second={tokens_per_second:.1f}"
        )
        del model

    if len(results) == 2:
        baseline, research = results
        print(
            "comparison_delta "
            f"validation_loss={research[2] - baseline[2]:+.6f} "
            f"parameter_delta={research[1] - baseline[1]:+d} "
            f"tokens_per_second_delta={research[4] - baseline[4]:+.1f}"
        )
        print("Repeat with multiple seeds before choosing a default architecture.")


if __name__ == "__main__":
    main()
