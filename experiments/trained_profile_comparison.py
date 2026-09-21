"""Train tiny and scaled profiles under the same budget and compare them.

The experiment keeps the scientific question narrow: does extra model capacity
change measured language-model learning when data, initialization, optimizer,
learning-rate schedule, clipping and update count are held fixed?

It intentionally does not modify generation with repetition penalties or other
heuristics. Those changes belong after the training/evaluation evidence.
"""

from experiments.compare_lm_profiles import (
    CONTEXT_LENGTH,
    SCALED,
    TINY,
    VALIDATION_FRACTION,
    VOCAB_SIZE,
)
from experiments.scaled_lm_diagnostics import CORPUS, evaluate
from src.gradient_clipping import clip_grad_norm_
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.schedulers import CosineAnnealing
from src.tokenizer import BPETokenizer

STEPS = 8
BATCH_SIZE = 4
LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 0.001
MAX_GRAD_NORM = 1.0
SEED = 7


def _build(config, corpus=CORPUS):
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
    train, validation = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    model = TinyLanguageModel(tokenizer.vocab_size, seed=SEED, **config)
    return model, tokenizer, train, validation


def train_profile(config, steps=STEPS, corpus=CORPUS):
    if steps <= 0:
        raise ValueError("steps must be positive")

    model, tokenizer, train, validation = _build(config, corpus)
    scheduler = CosineAnnealing(
        LEARNING_RATE,
        total_steps=steps,
        min_lr=MIN_LEARNING_RATE,
    )
    initial_train = evaluate(model, train, BATCH_SIZE)
    initial_validation = evaluate(model, validation, BATCH_SIZE)
    history = []

    for step in range(steps):
        batches = list(
            train.iter_batches(
                BATCH_SIZE,
                shuffle=True,
                seed=SEED + step,
            )
        )
        batch = batches[step % len(batches)]
        model.zero_grad()
        total_loss = None
        total_tokens = 0
        for inputs, targets in batch:
            item_loss = model.loss(inputs, targets)
            weighted = item_loss * len(targets)
            total_loss = weighted if total_loss is None else total_loss + weighted
            total_tokens += len(targets)
        loss = total_loss / total_tokens
        loss.backward()
        gradient_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        learning_rate = scheduler.get_lr(step)
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

        history.append(
            {
                "step": step + 1,
                "batch_loss": float(loss.data),
                "gradient_norm": gradient_norm,
                "learning_rate": learning_rate,
            }
        )

    final_train = evaluate(model, train, BATCH_SIZE)
    final_validation = evaluate(model, validation, BATCH_SIZE)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "history": history,
        "config": dict(config),
    }


def compare_trained(steps=STEPS, corpus=CORPUS):
    """Return matched-budget results for the tiny and scaled profiles."""
    return {
        "tiny": train_profile(TINY, steps=steps, corpus=corpus),
        "scaled": train_profile(SCALED, steps=steps, corpus=corpus),
    }


if __name__ == "__main__":
    report = compare_trained()
    for name, result in report.items():
        print(name)
        print("  initial train:", result["initial_train"])
        print("  final train:", result["final_train"])
        print("  initial validation:", result["initial_validation"])
        print("  final validation:", result["final_validation"])
        print("  last update:", result["history"][-1])
