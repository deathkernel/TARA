"""Compare tiny and scaled language models after the same training budget.

This experiment is intentionally measurement-first. Both models see the same
corpus, tokenizer, batch size, optimizer, learning-rate schedule, seed policy,
and number of updates. The comparison reports training/validation loss,
next-token accuracy, entropy, and scalar parameter count so capacity can be
separated from training progress.
"""

from experiments.scaled_lm_diagnostics import CORPUS, evaluate
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.gradient_clipping import clip_grad_norm_
from src.schedulers import CosineAnnealing
from src.tokenizer import BPETokenizer


PROFILES = {
    "tiny": {"embedding_dim": 8, "ff_dim": 16, "num_heads": 2, "num_layers": 1},
    "scaled": {"embedding_dim": 32, "ff_dim": 64, "num_heads": 4, "num_layers": 2},
}
VOCAB_SIZE = 64
CONTEXT_LENGTH = 64
VALIDATION_FRACTION = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.01
MIN_LEARNING_RATE = 0.001
MAX_GRAD_NORM = 1.0
SEED = 7


def parameter_count(model):
    """Count scalar trainable parameters exposed by the model."""
    return sum(1 for _ in model.parameters())


def build_profile(config, corpus=CORPUS):
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
    train, validation = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    model = TinyLanguageModel(tokenizer.vocab_size, seed=SEED, **config)
    return model, train, validation


def train_profile(config, steps, corpus=CORPUS):
    """Train one profile and return metrics before and after the budget."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    model, train_dataset, validation_dataset = build_profile(config, corpus)
    scheduler = CosineAnnealing(
        LEARNING_RATE,
        total_steps=steps,
        min_lr=MIN_LEARNING_RATE,
    )
    initial_train = evaluate(model, train_dataset, BATCH_SIZE)
    initial_validation = evaluate(model, validation_dataset, BATCH_SIZE)

    for step in range(steps):
        batches = list(
            train_dataset.iter_batches(
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
            loss = model.loss(inputs, targets)
            weighted = loss * len(targets)
            total_loss = weighted if total_loss is None else total_loss + weighted
            total_tokens += len(targets)
        loss = total_loss / total_tokens
        loss.backward()
        clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        learning_rate = scheduler.get_lr(step)
        for parameter in model.parameters():
            parameter.data -= learning_rate * parameter.grad

    return {
        "parameters": parameter_count(model),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": evaluate(model, train_dataset, BATCH_SIZE),
        "final_validation": evaluate(model, validation_dataset, BATCH_SIZE),
        "steps": steps,
    }


def compare_trained(steps=4, corpus=CORPUS):
    """Train both profiles for the same number of updates and compare them."""
    return {
        name: train_profile(config, steps=steps, corpus=corpus)
        for name, config in PROFILES.items()
    }


if __name__ == "__main__":
    report = compare_trained()
    for name, result in report.items():
        print(name, result)
