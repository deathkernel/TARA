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

def test_scheduler_extension_uses_new_total_step_budget():
    original = WarmupCosineScheduler(total_steps=2, warmup_steps=0, min_lr_ratio=0.2)
    extended = WarmupCosineScheduler(total_steps=4, warmup_steps=0, min_lr_ratio=0.2)

    assert original.multiplier(1) == pytest.approx(0.6)
    assert extended.multiplier(1) > original.multiplier(1)
    assert extended.multiplier(4) == pytest.approx(0.2)

@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WarmupCosineScheduler(10, warmup_steps=-1), "warmup_steps must be non-negative"),
        (lambda: WarmupCosineScheduler(10, min_lr_ratio=0.0), "min_lr_ratio must be in"),
        (lambda: EarlyStopping(patience=-1), "patience and min_delta must be non-negative"),
        (lambda: EarlyStopping(min_delta=-0.1), "patience and min_delta must be non-negative"),
    ],
)
def test_scheduler_and_early_stopping_reject_invalid_controls(factory, message):
    with pytest.raises(ValueError, match=message):
        factory()


def test_scheduler_caps_warmup_at_total_steps():
    scheduler = WarmupCosineScheduler(total_steps=3, warmup_steps=10, min_lr_ratio=0.2)

    assert scheduler.multiplier(0) == pytest.approx(1 / 3)
    assert scheduler.multiplier(2) == pytest.approx(1.0)
    assert scheduler.multiplier(3) == pytest.approx(0.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_early_stopping_rejects_non_finite_validation_values(value):
    stopper = EarlyStopping()
    with pytest.raises(ValueError, match="validation metric must be finite"):
        stopper.update(value)

