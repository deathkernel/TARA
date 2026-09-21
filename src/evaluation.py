"""Small scientific evaluation primitives for TARA.

Research basis:
- Reproducible ML experiments should separate measured results from the
  procedure and configuration that produced them.
- Regression testing compares a new measurement with a recorded baseline.
- Numerical checks such as finite-difference gradient checking validate an
  implementation independently of the training loop.

TARA keeps evaluation dependency-free and deterministic where possible. The
module records explicit metrics, compares them against tolerances, classifies
failures, and builds compact reports. It does not execute external actions.
"""

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class MetricResult:
    """One measured metric with an optional acceptable range."""

    name: str
    value: float
    minimum: float | None = None
    maximum: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("metric name must be a non-empty string")
        if not math.isfinite(float(self.value)):
            raise ValueError("metric value must be finite")
        if self.minimum is not None and not math.isfinite(float(self.minimum)):
            raise ValueError("metric minimum must be finite")
        if self.maximum is not None and not math.isfinite(float(self.maximum)):
            raise ValueError("metric maximum must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("metric minimum must not exceed maximum")

    @property
    def passed(self):
        return ((self.minimum is None or self.value >= self.minimum)
                and (self.maximum is None or self.value <= self.maximum))


def compare_metric(name, value, *, minimum=None, maximum=None):
    """Measure one scalar against an explicit acceptance interval."""
    return MetricResult(name, float(value), minimum, maximum)


def compare_baseline(name, value, baseline, *, tolerance=0.0, direction="lower"):
    """Compare a metric against a baseline using an explicit direction.

    ``direction='lower'`` means values must not increase beyond tolerance;
    ``direction='higher'`` means values must not decrease beyond tolerance.
    """
    if tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("tolerance must be finite and non-negative")
    if direction not in {"lower", "higher"}:
        raise ValueError("direction must be 'lower' or 'higher'")
    value = float(value)
    baseline = float(baseline)
    if not math.isfinite(value) or not math.isfinite(baseline):
        raise ValueError("value and baseline must be finite")
    if direction == "lower":
        passed = value <= baseline + tolerance
    else:
        passed = value >= baseline - tolerance
    return {
        "name": name,
        "value": value,
        "baseline": baseline,
        "tolerance": float(tolerance),
        "direction": direction,
        "passed": passed,
    }


def classify_failures(results):
    """Classify failed metric results without hiding the measured values."""
    failures = []
    for result in results:
        if not result.passed:
            failures.append({
                "name": result.name,
                "value": result.value,
                "minimum": result.minimum,
                "maximum": result.maximum,
                "reason": "below minimum" if result.minimum is not None and result.value < result.minimum
                          else "above maximum",
            })
    return failures


def build_report(results, metadata=None):
    """Build a JSON-serializable evaluation report."""
    results = list(results)
    failures = classify_failures(results)
    return {
        "metadata": dict(metadata or {}),
        "metrics": [asdict(result) | {"passed": result.passed} for result in results],
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }
