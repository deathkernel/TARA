from src.architecture_optimization import (
    ArchitectureComponent,
    ArchitectureMetrics,
    ArchitectureOptimizer,
    ArchitectureVariant,
    BottleneckDetector,
    fingerprint_variant,
)


def test_architecture_optimizer_promotes_measured_improvement():
    baseline = ArchitectureVariant(
        "base",
        (
            ArchitectureComponent("reasoner", True, (("width", 1.0),), critical=True),
            ArchitectureComponent("cache", True, (("size", 1.0),)),
        ),
    )

    def evaluate(variant):
        disabled_cache = not next(c for c in variant.components if c.name == "cache").enabled
        return ArchitectureMetrics(
            task_score=0.90 if disabled_cache else 0.88,
            robustness=0.95,
            latency_ms=50.0 if disabled_cache else 100.0,
            memory_mb=80.0 if disabled_cache else 100.0,
            throughput=2.0 if disabled_cache else 1.0,
        )

    optimizer = ArchitectureOptimizer(
        evaluator=evaluate,
        detector=BottleneckDetector(),
        latency_budget_ms=70.0,
        memory_budget_mb=200.0,
    )
    result = optimizer.optimize(baseline, rounds=2, candidates_per_round=2)
    assert result.final.variant.variant_id != "base"
    assert result.final.metrics.task_score >= 0.90
    assert result.rounds[0].decision.promoted


def test_fingerprint_is_stable():
    variant = ArchitectureVariant(
        "x",
        (ArchitectureComponent("a", parameters=(("p", 1.0),)),),
    )
    assert fingerprint_variant(variant) == fingerprint_variant(variant)


def test_budget_bottleneck_is_detected():
    metrics = ArchitectureMetrics(0.9, 0.9, 200.0, 50.0, 1.0)
    report = BottleneckDetector().diagnose(metrics, latency_budget_ms=100.0, memory_budget_mb=100.0)
    assert report.primary == "latency"
    assert report.severity > 0
