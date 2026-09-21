"""Matched-budget optimizer comparison for TARA's PC-friendly language model.

The experiment keeps model, corpus, batch size, update count, seed, gradient
clipping, and learning-rate schedule fixed. Only the optimizer changes.
This makes the comparison descriptive rather than declaring a universally
"best" optimizer.
"""

import math

from experiments.scaled_lm_diagnostics import (
    BATCH_SIZE if False else CORPUS,
    CONTEXT_LENGTH,
    EMBEDDING_DIM,
    FF_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    VALIDATION_FRACTION,
    VOCAB_SIZE,
    evaluate,
)
from src.gradient_clipping import clip_grad_norm_
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.optimizers import AdamW
from src.schedulers import CosineAnnealing
from src.tokenizer import BPETokenizer

STEPS = 40
BATCH_SIZE = 4
MAX_GRAD_NORM = 1.0
SEED = 7
ADAMW_LR = 0.001
ADAMW_WEIGHT_DECAY = 0.01
SGD_LR = 0.03


class SGD:
    """Minimal dependency-free SGD baseline for a matched experiment."""

    def __init__(self, parameters, learning_rate):
        self.parameters = list(parameters)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.learning_rate = float(learning_rate)
        self.step_count = 0

    def step(self, learning_rate=None):
        lr = self.learning_rate if learning_rate is None else float(learning_rate)
        if lr <= 0:
            raise ValueError("learning_rate must be positive")
        self.step_count += 1
        squared_norm = 0.0
        for parameter in self.parameters:
            gradient = float(parameter.grad)
            if not math.isfinite(gradient):
                raise ValueError("gradient must be finite")
            squared_norm += gradient * gradient
            parameter.data -= lr * gradient
        return math.sqrt(squared_norm)


def _build_data(corpus):
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
    train, validation = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    return tokenizer, train, validation


def _run(name, optimizer_kind, steps=STEPS, corpus=CORPUS):
    tokenizer, train, validation = _build_data(corpus)
    model = TinyLanguageModel(
        tokenizer.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        ff_dim=FF_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        seed=SEED,
    )

    if optimizer_kind == "AdamW":
        optimizer = AdamW(
            model.parameters(),
            learning_rate=ADAMW_LR,
            weight_decay=ADAMW_WEIGHT_DECAY,
        )
        scheduler = CosineAnnealing(ADAMW_LR, total_steps=steps, min_lr=ADAMW_LR * 0.1)
    elif optimizer_kind == "SGD":
        optimizer = SGD(model.parameters(), learning_rate=SGD_LR)
        scheduler = CosineAnnealing(SGD_LR, total_steps=steps, min_lr=SGD_LR * 0.1)
    else:
        raise ValueError("unknown optimizer")

    initial_train = evaluate(model, train, BATCH_SIZE)
    initial_validation = evaluate(model, validation, BATCH_SIZE)
    history = []

    for step in range(steps):
        batches = list(train.iter_batches(BATCH_SIZE, shuffle=True, seed=SEED + step))
        batch = batches[step % len(batches)]
        model.zero_grad()
        total_loss = None
        total_tokens = 0
        for inputs, targets in batch:
            loss = model.loss(inputs, targets)
            weighted = loss * len(targets)
            total_loss = weighted if total_loss is None else total_loss + weighted
            total_tokens += len(targets)
        loss = total_loss / total_tokens
        loss.backward()
        clipped_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        learning_rate = scheduler.get_lr(step)
        optimizer.step(learning_rate=learning_rate)
        history.append({
            "step": step,
            "loss": loss.data,
            "learning_rate": learning_rate,
            "gradient_norm": clipped_norm,
        })

    return {
        "optimizer": name,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": evaluate(model, train, BATCH_SIZE),
        "final_validation": evaluate(model, validation, BATCH_SIZE),
        "history": history,
        "steps": steps,
        "config": {
            "batch_size": BATCH_SIZE,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "model": "same TinyLanguageModel scaled profile",
        },
    }


def compare(steps=STEPS, corpus=CORPUS):
    """Run SGD and AdamW under the same update budget."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    return {
        "SGD": _run("SGD", "SGD", steps=steps, corpus=corpus),
        "AdamW": _run("AdamW", "AdamW", steps=steps, corpus=corpus),
    }


if __name__ == "__main__":
    for name, result in compare().items():
        print(name, result["initial_train"], "->", result["final_train"])
        print("validation:", result["initial_validation"], "->", result["final_validation"])
