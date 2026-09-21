"""Compare character and BPE tokenization under the same tiny LM budget.

Research-first experiment:
- keep TARA's Transformer architecture and optimizer unchanged;
- train separate models with character and BPE token IDs;
- compare validation loss and generated text;
- do not replace TARA's default tokenizer until this experiment provides
  evidence that BPE helps this small training setup.

This is a controlled educational experiment, not a benchmark against
production language models.
"""

from pathlib import Path
import random
import sys
import time

# Allow both `python -m experiments.tokenizer_lm_comparison` and the simpler
# `python experiments/tokenizer_lm_comparison.py` from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.language_model import TinyLanguageModel
from src.text_dataset import load_tinystories_text
from src.tokenizer import BPETokenizer, CharTokenizer


MAX_TRAIN_CHARS = 4096
MAX_VALIDATION_CHARS = 1024
BPE_VOCAB_SIZE = 64
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 1000
LEARNING_RATE = 0.01
CONTEXT_LENGTH = 16
SEED = 7
GRAD_CLIP = 1.0


class Adam:
    """Small Adam optimizer for TARA's scalar Value parameters."""

    def __init__(self, parameters):
        self.parameters = list(parameters)
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
            gradient = max(-GRAD_CLIP, min(GRAD_CLIP, parameter.grad))
            self.first_moment[i] = self.beta1 * self.first_moment[i] + (1.0 - self.beta1) * gradient
            self.second_moment[i] = self.beta2 * self.second_moment[i] + (1.0 - self.beta2) * gradient * gradient
            m_hat = self.first_moment[i] / bias1
            v_hat = self.second_moment[i] / bias2
            parameter.data -= LEARNING_RATE * m_hat / (v_hat ** 0.5 + self.epsilon)


def make_windows(ids):
    windows = []
    for start in range(0, len(ids) - 1, CONTEXT_LENGTH):
        window = ids[start:start + CONTEXT_LENGTH + 1]
        if len(window) >= 2:
            windows.append((window[:-1], window[1:]))
    if not windows:
        raise ValueError("dataset is too small for a training window")
    return windows


def train_and_evaluate(tokenizer, train_text, validation_text):
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        seed=SEED,
    )
    train_ids = tokenizer.encode(train_text)
    validation_ids = tokenizer.encode(validation_text)
    train_windows = make_windows(train_ids)
    validation_windows = make_windows(validation_ids)
    optimizer = Adam(model.parameters())
    rng = random.Random(SEED)

    start = time.perf_counter()
    for step in range(STEPS):
        inputs, targets = rng.choice(train_windows)
        model.zero_grad()
        loss = model.loss(inputs, targets)
        loss.backward()
        optimizer.step()
        if step % 100 == 0 or step == STEPS - 1:
            print(f"  step={step:4d} loss={loss.data:.6f}")
    elapsed = time.perf_counter() - start

    model.zero_grad()
    validation_loss = model.loss(
        validation_windows[0][0], validation_windows[0][1]
    ).data
    return model, validation_loss, elapsed


def main():
    train_text = load_tinystories_text(max_chars=MAX_TRAIN_CHARS, split="train")
    validation_text = load_tinystories_text(max_chars=MAX_VALIDATION_CHARS, split="validation")

    experiments = [
        ("character", CharTokenizer(train_text)),
        ("bpe", BPETokenizer(train_text, vocab_size=BPE_VOCAB_SIZE)),
    ]

    print("TARA tokenizer + language-model comparison")
    print(f"Train characters: {len(train_text)}")
    print(f"Validation characters: {len(validation_text)}")
    print(f"Steps per tokenizer: {STEPS}")

    for name, tokenizer in experiments:
        print(f"\n[{name}]")
        print(f"Vocabulary: {tokenizer.vocab_size}")
        print(f"Train tokens: {len(tokenizer.encode(train_text))}")
        print(f"Validation tokens: {len(tokenizer.encode(validation_text))}")
        model, validation_loss, elapsed = train_and_evaluate(
            tokenizer, train_text, validation_text
        )
        print(f"Validation loss (first validation window): {validation_loss:.6f}")
        print(f"Training time: {elapsed:.3f}s")
        prompt = validation_text[:20]
        ids = tokenizer.encode(prompt)
        for _ in range(60):
            context = ids[-CONTEXT_LENGTH:]
            ids.append(model.next_token(context))
        print("Greedy sample:")
        print(tokenizer.decode(ids))


if __name__ == "__main__":
    main()
