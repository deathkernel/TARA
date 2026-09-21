"""Measurement suite for TARA's PC-oriented scaling phase.

The experiment keeps comparisons scientifically interpretable: profiles use the
same corpus protocol, seed, optimizer family, schedule, clipping rule, batch
size and update budget. It reports capacity, optimization progress, held-out
behavior and a simple token/update compute proxy rather than assuming that
larger capacity is automatically better.
"""

import math

from experiments.compare_trained_lm_profiles import BATCH_SIZE, CONTEXT_LENGTH, PROFILES, SEED, train_profile
from experiments.scaled_lm_diagnostics import CORPUS, evaluate
from experiments.heldout_generalization import HELDOUT_CORPUS, build_heldout_dataset


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _summary(result, name, heldout):
    initial = result["initial_train"]
    final = result["final_train"]
    validation = result["final_validation"]
    history = result["history"]
    tokens_per_update = BATCH_SIZE * CONTEXT_LENGTH
    return {
        "profile": name,
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


def run_scaling_experiment(steps=4, corpus=CORPUS, heldout_text=HELDOUT_CORPUS):
    """Train both capacity profiles under a matched budget and evaluate on held-out text."""
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
            "heldout_corpus": "separate text encoded with each profile's training tokenizer",
        },
        "profiles": [],
    }

    for name, config in PROFILES.items():
        result = train_profile(config, steps=steps, corpus=corpus)
        heldout_dataset = build_heldout_dataset(
            result["tokenizer"],
            text=heldout_text,
            context_length=CONTEXT_LENGTH,
        )
        heldout = evaluate(result["model"], heldout_dataset, BATCH_SIZE)
        report["profiles"].append(_summary(result, name, heldout))
    return report


if __name__ == "__main__":
    print(run_scaling_experiment())
