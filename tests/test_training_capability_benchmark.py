import json

import torch

from src.training_capability_benchmark import benchmark_checkpoint
from src.training_pipeline import TrainingConfig, TrainingPipeline


def test_benchmark_detects_training_improvement(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps({"problem": "aaaa", "solution": "bbbb", "tests": "ok"}) + "\n"
        + json.dumps({"problem": "cccc", "solution": "dddd", "tests": "ok"}) + "\n",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "model.pt"
    config = TrainingConfig(
        steps=2,
        batch_size=1,
        context=4,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        num_layers=1,
        dropout=0.0,
        validation_split=0.5,
        seed=7,
        log_every=1,
        checkpoint_every=0,
    )
    TrainingPipeline(config, device="cpu").train(data=dataset, output=checkpoint)
    report = benchmark_checkpoint(checkpoint, dataset, device="cpu")

    assert report.baseline_validation_loss is not None
    assert report.trained_validation_loss is not None
    assert report.trained_validation_loss < report.baseline_validation_loss
    assert report.trained_better is True
    assert report.loss_delta < 0
