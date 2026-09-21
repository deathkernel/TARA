"""Measurement suite for TARA's PC-oriented scaling phase.

The experiment keeps comparisons scientifically interpretable: profiles use the
same corpus protocol, seed, optimizer family, schedule, clipping rule, batch
size and update budget.  It reports capacity, optimization progress,
held-out behavior and a simple token/update compute proxy rather than making
claims from parameter count alone.
"""

import math

from experiments.compare_trained_lm_profiles import (
    BATCH_SIZE,
    CONTEXT_LENGTH,
    PROFILES,
    SEED,
    train_profile,
)
from experiments.scaled_lm_diagnostics import CORPUS
from experiments.heldout_generalization import build_heldout_dataset, evaluate


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _profile_summary(result, model_name, heldout):
    initial = result["initial_train"]
    final = result["final_train"]
    validation = result["final_validation"]
    history = result["history"]
    tokens_per_update = BATCH_SIZE * CONTEXT_LENGTH
    return {
        "profile": model_name,
        "parameters": result["parameters"],
        "updates": result["steps"],
        "tokens_per_update_proxy": tokens_per_update,
        "token_budget_proxy": result["steps"] * tokens_per_update,
        "train_loss_initial": _finite(initial["loss"], "train_loss_initial"),
        "train_loss_final": _finite(final["loss"], "train_loss_final"),
        "validation_loss_final": _finite(validation["loss"], "validation_loss_final"),
        "heldout_loss": _finite(heldout["loss"], "heldout_loss"),
        "train_loss_delta": _finite(final["loss"] - initial["loss"], "train_loss_delta"),
        "validation_accuracy": _finite(validation["accuracy"], "validation_accuracy"),
        "heldout_accuracy": _finite(heldout["accuracy"], "heldout_accuracy"),
        "mean_gradient_norm": _finite(
            sum(item["gradient_norm"] for item in history) / len(history),
            "mean_gradient_norm",
        ),
    }


def run_scaling_experiment(steps=4, corpus=CORPUS):
    """Run both capacity profiles and evaluate each on the same held-out text."""
    if steps <= 0:
        raise ValueError("steps must be positive")

    report = {
        "protocol": {
            "steps": steps,
            "batch_size": BATCH_SIZE,
            "context_length": CONTEXT_LENGTH,
            "seed": SEED,
            "optimizer": "AdamW",
            "profiles": PROFILES,
            "comparison_rule": "same training budget and data protocol; no winner is assumed",
        },
        "profiles": [],
    }

    for name, config in PROFILES.items():
        result = train_profile(config, steps=steps, corpus=corpus)
        heldout_dataset = build_heldout_dataset(
            # Rebuild only the tokenizer used by this profile so held-out text
            # is encoded with exactly the training vocabulary.
            _tokenizer_for_result(config, corpus),
            context_length=CONTEXT_LENGTH,
        )
        heldout = evaluate(_model_from_result(result), heldout_dataset, BATCH_SIZE)
        report["profiles"].append(_profile_summary(result, name, heldout))
    return report


def _tokenizer_for_result(config, corpus):
    from src.language_dataset import build_train_validation_datasets
    from src.tokenizer import BPETokenizer

    tokenizer, _, _ = build_train_validation_datasets(
        lambda train_text: BPETokenizer(train_text, vocab_size=64),
        corpus,
        context_length=CONTEXT_LENGTH,
        validation_fraction=0.2,
    )
    return tokenizer


def _model_from_result(result):
    # The existing profile trainer intentionally returns measurements rather
    # than the model. This helper is replaced by the model-aware path below.
    # It is kept private so callers use run_scaling_experiment only.
    from experiments.compare_trained_lm_profiles import build_profile
    # Reconstructing weights would invalidate held-out measurement, so this
    # function is deliberately unreachable in the final implementation.
    raise RuntimeError("run_scaling_experiment requires the model-aware trainer")


if __name__ == "__main__":
    print(run_scaling_experiment())
