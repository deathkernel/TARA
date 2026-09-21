"""Matched-budget trained comparison of TARA's tiny and scaled profiles.

The experiment isolates model capacity by keeping the corpus, tokenizer,
training updates, batch size, optimizer, scheduler, and clipping fixed.
Both profiles therefore receive the same number of parameter-update steps.

Results are descriptive: a larger model is not assumed to be better before
measurement. This script reports the evidence needed for that decision.
"""

import math

from experiments.scaled_lm_diagnostics import CORPUS, evaluate
from src.gradient_clipping import clip_grad_norm_
from src.language_dataset import build_causal_datasets
from src.language_model import TinyLanguageModel
from src.optimizers import AdamW
from src.schedulers import CosineAnnealing
from src.tokenizer import BPETokenizer

TINY = {"embedding_dim": 8, "ff_dim": 16, "num_heads": 2, "num_layers": 1}
SCALED = {"embedding_dim": 32, "ff_dim": 64, "num_heads": 4, "num_layers": 2}
VOCAB_SIZE = 64
CONTEXT_LENGTH = 64
VALIDATION_FRACTION = 0.2
STEPS = 40
BATCH_SIZE = 4
LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 0.0001
MAX_GRAD_NORM = 1.0
SEED = 7


def parameter_count(model):
    return sum(1 for _ in model.parameters())


def _build_data(corpus):
    tokenizer = BPETokenizer(corpus, vocab_size=VOCAB_SIZE)
    train, validation = build_causal_datasets(
        tokenizer,
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=VALIDATION_FRACTION,
    )
    return tokenizer, train, validation


def train_profile(config, steps=STEPS, corpus=CORPUS, seed=SEED):
    """Train one profile under the shared experimental budget."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    tokenizer, train, validation = _build_data(corpus)
    model = TinyLanguageModel(tokenizer.vocab_size, seed=seed, **config)
    optimizer = AdamW(model.parameters(), learning_rate=LEARNING_RATE, weight_decay=0.0)
    scheduler = CosineAnnealing(
        LEARNING_RATE,
        total_steps=steps,
        min_lr=MIN_LEARNING_RATE,
    )

    initial_train = evaluate(model, train, BATCH_SIZE)
    initial_validation = evaluate(model, validation, BATCH_SIZE)
    history = []

    for step in range(steps):
        batches = list(train.iter_batches(BATCH_SIZE, shuffle=True, seed=seed + step))
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
        gradient_norm = clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        learning_rate = scheduler.get_lr(step)
        optimizer.step(learning_rate=learning_rate)
        history.append({
            "step": step,
            "loss": loss.data,
            "learning_rate": learning_rate,
            "gradient_norm": gradient_norm,
        })

    return {
        "parameters": parameter_count(model),
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": evaluate(model, train, BATCH_SIZE),
        "final_validation": evaluate(model, validation, BATCH_SIZE),
        "history": history,
        "config": {
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": seed,
            "optimizer": "AdamW",
            "weight_decay": 0.0,
        },
    }


def compare(steps=STEPS, corpus=CORPUS, seed=SEED):
    """Train tiny and scaled profiles under one controlled budget."""
    return {
        "tiny": train_profile(TINY, steps=steps, corpus=corpus, seed=seed),
        "scaled": train_profile(SCALED, steps=steps, corpus=corpus, seed=seed),
    }


def validate_result(result):
    """Validate that the experiment produced finite, comparable evidence."""
    for name, profile in result.items():
        assert name in {"tiny", "scaled"}
        assert profile["parameters"] > 0
        for split in ("initial_train", "initial_validation", "final_train", "final_validation"):
            metrics = profile[split]
            assert math.isfinite(metrics["loss"])
            assert 0.0 <= metrics["accuracy"] <= 1.0
            assert math.isfinite(metrics["entropy"])
            assert metrics["entropy"] >= 0.0
            assert metrics["tokens"] > 0
        assert len(profile["history"]) == profile["config"]["steps"]
        assert all(math.isfinite(item["loss"]) for item in profile["history"])
        assert all(math.isfinite(item["gradient_norm"]) for item in profile["history"])


if __name__ == "__main__":
    report = compare()
    validate_result(report)
    for name, result in report.items():
        print(name)
        print("parameters:", result["parameters"])
        print("train:", result["initial_train"], "->", result["final_train"])
        print("validation:", result["initial_validation"], "->", result["final_validation"])
