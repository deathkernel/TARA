import math

from experiments.scaled_lm_diagnostics import build_experiment, evaluate


def test_scaled_experiment_builds_with_bpe_and_expected_capacity():
    model, tokenizer, train_dataset, validation_dataset = build_experiment()
    assert tokenizer.vocab_size <= 64
    assert model.transformer.num_layers == 2
    assert model.transformer.blocks[0].num_heads == 4
    assert model.embedding.embedding_dim == 32
    assert train_dataset.examples
    assert validation_dataset.examples


def test_diagnostics_are_finite_and_have_valid_ranges():
    model, _, train_dataset, validation_dataset = build_experiment()
    for dataset in (train_dataset, validation_dataset):
        metrics = evaluate(model, dataset)
        assert math.isfinite(metrics["loss"])
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert math.isfinite(metrics["entropy"])
        assert metrics["entropy"] >= 0.0
        assert metrics["tokens"] > 0
