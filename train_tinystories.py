"""Train and evaluate TARA on TinyStories with a faster learning loop.

The model is still TARA's own tiny Transformer and scalar autodiff engine.
This trainer changes the optimization strategy rather than the architecture:
- a larger text slice gives the model more varied training signal;
- stochastic context sampling avoids rebuilding the whole dataset graph every step;
- Adam adapts each parameter's step size and usually reaches useful loss faster;
- gradient clipping keeps tiny experiments stable.

This is deliberately a small-PC experiment, not a claim of reproducing a
production-scale language model.
"""

import random
import time

from src.language_model import TinyLanguageModel
from src.text_dataset import load_tinystories_text
from src.tokenizer import CharTokenizer


MAX_TRAIN_CHARS = 4096
MAX_VALIDATION_CHARS = 1024
EMBEDDING_DIM = 3
FF_DIM = 6
STEPS = 2000
LEARNING_RATE = 0.01
CONTEXT_LENGTH = 16
SEED = 7
GRAD_CLIP = 1.0


class Adam:
    """Small Adam optimizer for TARA's scalar Value parameters."""

    def __init__(self, parameters, learning_rate=LEARNING_RATE):
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
            gradient = max(-GRAD_CLIP, min(GRAD_CLIP, parameter.grad))
            self.first_moment[i] = (
                self.beta1 * self.first_moment[i] + (1.0 - self.beta1) * gradient
            )
            self.second_moment[i] = (
                self.beta2 * self.second_moment[i] + (1.0 - self.beta2) * gradient * gradient
            )
            m_hat = self.first_moment[i] / bias1
            v_hat = self.second_moment[i] / bias2
            parameter.data -= self.learning_rate * m_hat / (v_hat ** 0.5 + self.epsilon)


def make_windows(ids):
    """Precompute candidate context/target windows once."""
    if len(ids) < 2:
        raise ValueError("dataset must contain at least two tokens")
    windows = []
    for start in range(0, len(ids) - 1, CONTEXT_LENGTH):
        window = ids[start:start + CONTEXT_LENGTH + 1]
        if len(window) >= 2:
            windows.append((window[:-1], window[1:]))
    if not windows:
        raise ValueError("dataset must contain at least one target token")
    return windows


def token_loss(model, windows):
    """Return token-weighted next-token loss over supplied windows."""
    total_loss = None
    total_tokens = 0
    for inputs, targets in windows:
        loss = model.loss(inputs, targets)
        weighted_loss = loss * len(targets)
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        total_tokens += len(targets)
    return total_loss / total_tokens


def train(train_corpus=None, steps=STEPS, learning_rate=LEARNING_RATE):
    if train_corpus is None:
        train_corpus = load_tinystories_text(
            max_chars=MAX_TRAIN_CHARS, split="train"
        )

    tokenizer = CharTokenizer(train_corpus)
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        seed=SEED,
    )
    train_ids = tokenizer.encode(train_corpus)
    windows = make_windows(train_ids)
    parameters = model.parameters()
    optimizer = Adam(parameters, learning_rate=learning_rate)
    rng = random.Random(SEED)

    start_time = time.perf_counter()
    running_loss = 0.0
    for step in range(steps):
        # One random context per optimizer step is dramatically cheaper than
        # rebuilding a graph over the entire text on every step.
        batch = [rng.choice(windows)]
        model.zero_grad()
        loss = token_loss(model, batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.data

        if step % 100 == 0 or step == steps - 1:
            average = running_loss / (step % 100 + 1) if step % 100 else loss.data
            print(f"step={step:4d} train_loss={loss.data:.6f} window_avg={average:.6f}")
            running_loss = 0.0

    elapsed = time.perf_counter() - start_time
    print(f"Training time: {elapsed:.3f}s ({elapsed / steps:.4f}s/step)")
    return model, tokenizer, train_corpus


def evaluate(model, tokenizer, validation_corpus):
    """Measure next-token loss on unseen validation text."""
    validation_ids = tokenizer.encode(validation_corpus)
    windows = make_windows(validation_ids)
    model.zero_grad()
    loss = token_loss(model, windows)
    return loss.data


def generate(model, tokenizer, prompt, length=80, temperature=None, top_k=None, seed=0):
    """Generate text with greedy decoding or controlled sampling."""
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt must not be empty")

    rng = random.Random(seed)
    for _ in range(length):
        context = ids[-CONTEXT_LENGTH:]
        if temperature is None:
            next_id = model.next_token(context)
        else:
            next_id = model.sample_next_token(
                context, temperature=temperature, top_k=top_k, rng=rng
            )
        ids.append(next_id)
    return tokenizer.decode(ids)


if __name__ == "__main__":
    model, tokenizer, train_corpus = train()
    validation_corpus = load_tinystories_text(
        max_chars=MAX_VALIDATION_CHARS, split="validation"
    )
    validation_loss = evaluate(model, tokenizer, validation_corpus)
    prompt = validation_corpus[:20]

    print(f"\nTrain characters: {len(train_corpus)}")
    print(f"Validation characters: {len(validation_corpus)}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Validation loss: {validation_loss:.6f}")

    print("\nGreedy generation:")
    print(generate(model, tokenizer, prompt))

    print("\nTemperature=0.8, top-k=5 generation:")
    print(generate(model, tokenizer, prompt, temperature=0.8, top_k=5, seed=42))

    print("\nTemperature=1.0, top-k=5 generation:")
    print(generate(model, tokenizer, prompt, temperature=1.0, top_k=5, seed=42))
