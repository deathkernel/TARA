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

    assert scheduler.multiplier(0) == pytest.approx(0.5)
    assert scheduler.multiplier(2) == pytest.approx(1.0)
    assert scheduler.multiplier(3) == pytest.approx(0.2)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_early_stopping_rejects_non_finite_validation_values(value):
    stopper = EarlyStopping()
    with pytest.raises(ValueError, match="validation metric must be finite"):
        stopper.update(value)

@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('{"step": 1, "train_loss": 1.0, "validation_loss": null, "learning_rate": 0.1}\\n{"step": 1, "train_loss": 0.9, "validation_loss": null, "learning_rate": 0.1}\\n', "strictly increasing"),
        ('{"step": 0, "train_loss": 1.0, "validation_loss": null, "learning_rate": 0.1}\\n', "step must be positive"),
        ('{"step": 1, "train_loss": 1.0, "validation_loss": null, "learning_rate": 0.1}\\nnot-json\\n', "invalid metrics JSONL record at line 2"),
    ],
)
def test_tracker_rejects_invalid_or_non_monotonic_metrics(tmp_path, content, message):
    path = tmp_path / "metrics.jsonl"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        ExperimentTracker(path).metrics()


def test_tracker_accepts_strictly_increasing_metrics(tmp_path):
    tracker = ExperimentTracker(tmp_path / "metrics.jsonl")
    tracker.log(TrainingMetric(1, 1.0, None, 0.1))
    tracker.log(TrainingMetric(3, 0.8, None, 0.05))

    assert [metric.step for metric in tracker.metrics()] == [1, 3]

@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0, 1.0, None, 0.1), "metric step must be a positive integer"),
        ((1, float("nan"), None, 0.1), "train_loss must be finite"),
        ((1, 1.0, float("inf"), 0.1), "validation_loss must be finite"),
        ((1, 1.0, None, float("nan")), "learning_rate must be finite"),
        ((1, 1.0, None, -0.1), "learning_rate must be finite and non-negative"),
    ],
)
def test_training_metric_rejects_invalid_values(args, message):
    with pytest.raises(ValueError, match=message):
        TrainingMetric(*args)


def test_training_metric_allows_missing_validation_loss():
    metric = TrainingMetric(1, 1.0, None, 0.0)

    assert metric.validation_loss is None

@pytest.mark.parametrize(
    "record",
    [
        '{"step": 1, "train_loss": 1.0}',
        '{"step": 1, "train_loss": 1.0, "validation_loss": null, "learning_rate": 0.1, "extra": 1}',
        '[]',
        '{"step": 1, "train_loss": NaN, "validation_loss": null, "learning_rate": 0.1}',
    ],
)
def test_tracker_rejects_malformed_metric_schema(tmp_path, record):
    path = tmp_path / "metrics.jsonl"
    path.write_text(record + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid metrics JSONL record at line 1"):
        ExperimentTracker(path).metrics()


def test_tracker_log_rejects_duplicate_or_decreasing_steps(tmp_path):
    tracker = ExperimentTracker(tmp_path / "metrics.jsonl")
    tracker.log(TrainingMetric(2, 1.0, None, 0.1))

    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.log(TrainingMetric(2, 0.9, None, 0.05))
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.log(TrainingMetric(1, 0.9, None, 0.05))

    assert [metric.step for metric in tracker.metrics()] == [2]

