"""Simple two-command interface for TARA.

Usage:
    python tara.py train data/curriculum_v2.jsonl
    python tara.py chat

The CLI intentionally hides model/training knobs.  Training uses the supplied
dataset, a held-out validation split, validation token accuracy, and early
stopping.  Chat loads the resulting local checkpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.model_runtime import load_checkpoint
from src.training_pipeline import TrainingConfig, TrainingPipeline

CHECKPOINT = Path("checkpoints/tara.pt")
DEFAULT_MAX_NEW_TOKENS = 180
GENERATION_TEMPERATURE = 0.65
GENERATION_TOP_K = 40
GENERATION_REPETITION_PENALTY = 1.08
ROLE_MARKERS = ("<|user|>", "<|assistant|>")


def _training_config() -> TrainingConfig:
    """Use a low-heat communication model for gradual local training."""
    if torch.cuda.is_available():
        return TrainingConfig(
            steps=900,
            batch_size=4,
            context=128,
            embedding_dim=128,
            ff_dim=256,
            heads=4,
            num_layers=4,
            tokenizer="bpe",
            vocab_size=2048,
            lr=3e-4,
            validation_split=0.1,
            log_every=25,
            checkpoint_every=225,
            early_stopping_patience=8,
        )
    return TrainingConfig(
        steps=2000,
        batch_size=2,
        context=64,
        embedding_dim=128,
        ff_dim=256,
        heads=4,
        num_layers=4,
        tokenizer="bpe",
        vocab_size=2048,
        lr=3e-4,
        validation_split=0.15,
        log_every=25,
        checkpoint_every=400,
        early_stopping_patience=10,
        target_validation_accuracy=None,
    )


def train(dataset: str) -> int:
    dataset_path = Path(dataset)
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}", file=sys.stderr)
        return 2

    print(f"TARA training from: {dataset_path}")
    print(f"Checkpoint: {CHECKPOINT}")
    print("Training policy: validation loss + validation token accuracy + early stopping")
    print("Tokenizer: BPE")
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
    """Keep only the assistant's generated turn and remove role leakage."""
    if "<|assistant|>" in text:
        text = text.rsplit("<|assistant|>", 1)[-1]
    for marker in ROLE_MARKERS:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip()


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    """Generate only new assistant tokens, stopping at the next role marker."""
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        return ""

    prompt_ids = prompt_ids[-model.max_context:]
    generated: list[int] = []
    context = list(prompt_ids)
    vocab_size = tokenizer.vocab_size

    with torch.no_grad():
        for _ in range(max_new_tokens):
            x = torch.tensor([context[-model.max_context:]], dtype=torch.long)
            logits = model(x)[0, -1].clone()

            if generated:
                for token_id in set(generated[-64:]):
                    if 0 <= token_id < logits.numel():
                        if logits[token_id] > 0:
                            logits[token_id] /= GENERATION_REPETITION_PENALTY
                        else:
                            logits[token_id] *= GENERATION_REPETITION_PENALTY

            logits = logits / GENERATION_TEMPERATURE
            top_k = min(GENERATION_TOP_K, vocab_size, logits.numel())
            values, indices = torch.topk(logits, top_k)
            probabilities = torch.softmax(values, dim=-1)
            next_id = int(indices[torch.multinomial(probabilities, 1)].item())

            generated.append(next_id)
            context.append(next_id)

            partial = tokenizer.decode(generated)
            if any(marker in partial for marker in ROLE_MARKERS):
                break

    return _clean_response(tokenizer.decode(generated))


def chat() -> int:
    if not CHECKPOINT.exists():
        print("No trained TARA checkpoint found.")
        print("Run: python tara.py train data/curriculum_v2.jsonl")
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

        prompt = "".join(history)
        encoded = tokenizer.encode(prompt)
        if len(encoded) > model.max_context:
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
