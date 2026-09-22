import json

import pytest

from src.training_hardening import EarlyStopping, ExperimentTracker, TrainingControls, TrainingMetric, WarmupCosineScheduler


def test_controls_validate():
    with pytest.raises(ValueError):
        TrainingControls(gradient_accumulation_steps=0)


def test_warmup_cosine_is_bounded_and_decays():
    scheduler = WarmupCosineScheduler(100, warmup_steps=10, min_lr_ratio=0.2)
    assert scheduler.multiplier(0) < scheduler.multiplier(9)
    assert scheduler.multiplier(100) == pytest.approx(0.2)


def test_early_stopping_tracks_improvement_and_patience():
    stop = EarlyStopping(patience=1, min_delta=0.01)
    assert stop.update(1.0).improved
    assert not stop.update(1.0).should_stop
    assert stop.update(1.0).should_stop


def test_tracker_is_append_only_and_fingerprinted(tmp_path):
    path = tmp_path / "metrics.jsonl"
    tracker = ExperimentTracker(path)
    tracker.log(TrainingMetric(1, 2.0, 2.5, 0.001))
    tracker.log(TrainingMetric(2, 1.5, 2.0, 0.0009))
    metrics = tracker.metrics()
    assert [m.step for m in metrics] == [1, 2]
    lines = path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["step"] == 1
    assert json.loads(lines[1])["step"] == 2
    assert len(tracker.fingerprint()) == 64
