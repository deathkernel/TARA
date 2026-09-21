import pytest

from src.autograd import Value
from src.evaluation import MetricResult, build_report, classify_failures, compare_baseline, compare_metric, run_gradient_check


def test_metric_passes_inside_explicit_interval():
    result = compare_metric("accuracy", 0.9, minimum=0.8, maximum=1.0)
    assert result.passed


def test_metric_rejects_invalid_range_and_nonfinite_values():
    with pytest.raises(ValueError):
        MetricResult("x", 1.0, minimum=2.0, maximum=1.0)
    with pytest.raises(ValueError):
        MetricResult("x", float("nan"))


def test_baseline_comparison_respects_direction_and_tolerance():
    assert compare_baseline("loss", 0.51, 0.50, tolerance=0.02, direction="lower")["passed"]
    assert not compare_baseline("loss", 0.53, 0.50, tolerance=0.02, direction="lower")["passed"]
    assert compare_baseline("accuracy", 0.79, 0.80, tolerance=0.02, direction="higher")["passed"]
    assert not compare_baseline("accuracy", 0.77, 0.80, tolerance=0.02, direction="higher")["passed"]


def test_baseline_rejects_invalid_controls():
    with pytest.raises(ValueError):
        compare_baseline("x", 1, 1, tolerance=-1)
    with pytest.raises(ValueError):
        compare_baseline("x", 1, 1, direction="equal")


def test_failure_classification_and_report_preserve_measurements():
    results = [
        compare_metric("loss", 0.7, maximum=0.5),
        compare_metric("accuracy", 0.9, minimum=0.8),
    ]
    failures = classify_failures(results)
    assert failures == [{
        "name": "loss",
        "value": 0.7,
        "minimum": None,
        "maximum": 0.5,
        "reason": "above maximum",
    }]
    report = build_report(results, {"seed": 7})
    assert report["passed"] is False
    assert report["failure_count"] == 1
    assert report["metadata"] == {"seed": 7}


def test_run_gradient_check_records_numerical_evidence():
    parameter = Value(2.0)

    def loss_fn():
        return parameter * parameter

    loss = loss_fn()
    loss.backward()
    report = run_gradient_check(loss_fn, [parameter], epsilon=1e-6, tolerance=1e-5)
    assert report["passed"] is True
    assert report["parameter_count"] == 1
    assert report["max_relative_error"] < 1e-5
    assert report["diagnostics"][0]["analytic"] == pytest.approx(4.0)
