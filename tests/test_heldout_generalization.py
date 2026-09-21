import math

from experiments.heldout_generalization import build_heldout_dataset, HELDOUT_CORPUS
from experiments.scaled_lm_diagnostics import build_experiment, evaluate


def test_heldout_dataset_uses_fixed_tokenizer_and_contains_targets():
    model, tokenizer, _, _ = build_experiment()
    dataset = build_heldout_dataset(tokenizer, HELDOUT_CORPUS)
    assert len(dataset) > 0
    assert len(dataset[0][0]) == len(dataset[0][1])


def test_heldout_metrics_are_finite():
    model, tokenizer, _, _ = build_experiment()
    dataset = build_heldout_dataset(tokenizer, HELDOUT_CORPUS)
    metrics = evaluate(model, dataset)
    assert math.isfinite(metrics["loss"])
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert math.isfinite(metrics["entropy"])
    assert metrics["tokens"] > 0
