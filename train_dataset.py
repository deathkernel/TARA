"""Train TARA's tiny language model on any registered text dataset.

Examples:
    python train_dataset.py --dataset tinystories --tokenizer char
    python train_dataset.py --dataset wikitext2 --tokenizer bpe --vocab-size 128

The defaults stay deliberately small. Increase budgets only after measuring
stability and validation performance.
"""

import argparse
import random
import time

from src.language_model import TinyLanguageModel
from src.text_dataset import list_datasets, load_dataset_text
from src.tokenizer import BPETokenizer, CharTokenizer


DEFAULTS = {
    "train_chars": 4096,
    "validation_chars": 1024,
    "embedding_dim": 4,
    "ff_dim": 8,
    "steps": 500,
    "learning_rate": 0.01,
    "context_length": 16,
    "seed": 7,
}


class Adam:
    def __init__(self, parameters, learning_rate):
        self.parameters = list(parameters)
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.step_count = 0
        self.first_moment = [0.0] * len(self.parameters)
        self.second_moment = [0.0] * len(self.parameters)

    def step(self):
        self.step_count += 1
        bias1 = 1.0 - self.beta1 ** self.step_count
        bias2 = 1.0 - self.beta2 ** self.step_count

        for i, parameter in enumerate(self.parameters):
            gradient = max(-1.0, min(1.0, parameter.grad))
            self.first_moment[i] = (
                self.beta1 * self.first_moment[i] +
                (1.0 - self.beta1) * gradient
            )
            self.second_moment[i] = (
                self.beta2 * self.second_moment[i] +
                (1.0 - self.beta2) * gradient * gradient
            )

            m_hat = self.first_moment[i] / bias1
            v_hat = self.second_moment[i] / bias2
            parameter.data -= (
                self.learning_rate * m_hat /
                (v_hat ** 0.5 + self.epsilon)
            )


def make_windows(ids, context_length):
    """Precompute deterministic context/target windows."""
    if len(ids) < 2:
        raise ValueError("dataset must contain at least two tokens")

    windows = []
    for start in range(0, len(ids) - 1, context_length):
        window = ids[start:start + context_length + 1]
        if len(window) >= 2:
            windows.append((window[:-1], window[1:]))

    if not windows:
        raise ValueError("dataset is too short for training")
    return windows


def token_loss(model, windows):
    """Return token-weighted loss across a set of windows."""
    total = None
    token_count = 0

    for inputs, targets in windows:
        loss = model.loss(inputs, targets)
        weighted = loss * len(targets)
        total = weighted if total is None else total + weighted
        token_count += len(targets)

    return total / token_count


def build_tokenizer(text, kind, vocab_size):
    if kind == "char":
        return CharTokenizer(text)
    return BPETokenizer(text, vocab_size=vocab_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list_datasets(), default="tinystories")
    parser.add_argument("--tokenizer", choices=("char", "bpe"), default="char")
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=DEFAULTS["steps"])
    parser.add_argument("--train-chars", type=int, default=DEFAULTS["train_chars"])
    parser.add_argument(
        "--validation-chars",
        type=int,
        default=DEFAULTS["validation_chars"],
    )
    args = parser.parse_args()

    train_text = load_dataset_text(args.dataset, args.train_chars, "train")
    validation_text = load_dataset_text(
        args.dataset, args.validation_chars, "validation"
    )

    tokenizer = build_tokenizer(
        train_text,
        args.tokenizer,
        args.vocab_size,
    )

    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=DEFAULTS["embedding_dim"],
        ff_dim=DEFAULTS["ff_dim"],
        seed=DEFAULTS["seed"],
    )

    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)

    train_windows = make_windows(train_ids, DEFAULTS["context_length"])
    validation_windows = make_windows(
        validation_ids,
        DEFAULTS["context_length"],
    )

    optimizer = Adam(model.parameters(), DEFAULTS["learning_rate"])
    rng = random.Random(DEFAULTS["seed"])

    started = time.perf_counter()

    for step in range(args.steps):
        batch = [rng.choice(train_windows)]
        model.zero_grad()
        loss = token_loss(model, batch)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == args.steps - 1:
            print(f"step={step:4d} loss={loss.data:.6f}")

    elapsed = time.perf_counter() - started

    model.zero_grad()
    validation_loss = token_loss(model, validation_windows)

    print()
    print(f"Dataset: {args.dataset}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Vocabulary: {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_ids)}")
    print(f"Validation tokens: {len(validation_ids)}")
    print(f"Validation loss: {validation_loss.data:.6f}")
    print(f"Training time: {elapsed:.3f}s")
    print(f"Seconds/step: {elapsed / args.steps:.4f}")


if __name__ == "__main__":
    main()
