"""Simple two-command interface for TARA.

Usage:
    python tara.py train data/conversation_v1.jsonl
    python tara.py chat

The CLI intentionally hides model/training knobs.  Training uses the supplied
dataset, a held-out validation split, validation token accuracy, and early
stopping.  Chat loads the resulting local checkpoint.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

from src.model_runtime import load_checkpoint
from src.training_pipeline import TrainingConfig, TrainingPipeline

CHECKPOINT = Path("checkpoints/tara.pt")
DEFAULT_MAX_NEW_TOKENS = 180


def _training_config() -> TrainingConfig:
    """Return the single supported training policy for the TARA CLI."""
    if torch.cuda.is_available():
        return TrainingConfig(
            steps=1500,
            batch_size=16,
            context=128,
            embedding_dim=128,
            ff_dim=256,
            heads=4,
            num_layers=4,
            tokenizer="char",
            vocab_size=512,
            lr=3e-4,
            validation_split=0.1,
            log_every=25,
            checkpoint_every=250,
            early_stopping_patience=8,
        )
    return TrainingConfig(
        steps=1000,
        batch_size=8,
        context=128,
        embedding_dim=96,
        ff_dim=192,
        heads=4,
        num_layers=2,
        tokenizer="char",
        vocab_size=512,
        lr=3e-4,
        validation_split=0.1,
        log_every=10,
        checkpoint_every=100,
        early_stopping_patience=8,
    )


def train(dataset: str) -> int:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    print(f"TARA training from: {dataset_path}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Training policy: validation loss + validation token accuracy + early stopping")
    print("Tokenizer: character-level (fast and deterministic)")
    print("")

    summary = TrainingPipeline(_training_config()).train(
        dataset_path,
        CHECKPOINT,
    )
    print("")
    print("Training complete.")
    print(f"checkpoint={summary.checkpoint}")
    print(f"step={summary.final_step}")
    print(f"train_loss={summary.train_loss:.4f}")
    if summary.validation_loss is not None:
        print(f"validation_loss={summary.validation_loss:.4f}")
    if summary.validation_accuracy is not None:
        print(f"validation_accuracy={summary.validation_accuracy:.2%}")
    print(f"device={summary.device}")
    print(f"stopped_early={summary.stopped_early}")
    return 0


def _clean_response(text: str) -> str:
    """Keep only the assistant's generated turn."""
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
    for marker in ("<|user|>", "<|assistant|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    ids = tokenizer.encode(prompt)
    if not ids:
        return ""

    # Keep the model input inside its context window.
    ids = ids[-model.max_context:]
    generated = list(ids)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[-model.max_context:]
            x = torch.tensor([context], dtype=torch.long)
            logits = model(x)[0, -1]
            # Conservative sampling gives a small local model a better chance
            # of staying on-topic than unrestricted sampling.
            temperature = 0.75
            logits = logits / temperature
            top_k = min(40, logits.numel())
            values, indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            next_id = indices[torch.multinomial(probabilities, 1)].item()
            generated.append(int(next_id))

            decoded = tokenizer.decode(generated)
            if "<|user|>" in decoded or decoded.endswith("\n"):
                # A newline is a natural stopping point for the small
                # conversation model; the marker check prevents turn leakage.
                if "<|user|>" in decoded:
                    break

    return _clean_response(tokenizer.decode(generated))


def chat() -> int:
    if not CHECKPOINT.exists():
        print("No trained TARA checkpoint found.")
        print("Run: python tara.py train data/conversation_v1.jsonl")
        return 2

    model, tokenizer = load_checkpoint(CHECKPOINT)
    history: list[str] = []

    print("TARA ready. Type 'exit' to stop.")
    print("")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return 0

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            return 0

        history.append(f"<|user|>\n{user}\n<|assistant|>\n")

        # The model has a 128-token context by default. Keep the newest turns
        # so chat never exceeds its context window.
        prompt = "".join(history)
        encoded = tokenizer.encode(prompt)
        if len(encoded) > model.max_context:
            # Rebuild from newest complete turns rather than cutting through
            # a UTF-8/role boundary.
            kept: list[str] = []
            total = 0
            for turn in reversed(history):
                turn_len = len(tokenizer.encode(turn))
                if kept and total + turn_len > model.max_context - 8:
                    break
                kept.append(turn)
                total += turn_len
            history = list(reversed(kept))
            prompt = "".join(history)

        response = _generate(model, tokenizer, prompt)
        if not response:
            response = "I don't have a good response yet."

        print(f"TARA: {response}")
        history[-1] = history[-1] + response + "\n"
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="tara.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train TARA from a dataset")
    train_parser.add_argument("dataset", help="path to a JSONL/JSON/TXT training dataset")

    subparsers.add_parser("chat", help="chat with the trained TARA model")
    args = parser.parse_args()

    if args.command == "train":
        return train(args.dataset)
    return chat()


if __name__ == "__main__":
    raise SystemExit(main())
