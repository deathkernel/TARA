from src.scientific_experiment import (
    ExperimentDesigner,
    ExperimentVariable,
    ReplicationChecker,
    ScientificExperimentEngine,
)


def test_design_and_repeated_measurements_are_reproducible():
    designer = ExperimentDesigner()
    design = designer.design(
        "Higher x improves score",
        [ExperimentVariable("x", (0, 1))],
        repetitions=4,
        seed=7,
    )

    def run(params, seed):
        return {"score": 1.0 + params["x"] + (seed % 3) * 0.01}

    engine = ScientificExperimentEngine(bootstrap_samples=200)
    first = engine.run(design, run)
    second = engine.run(design, run)
    assert len(first.trials) == 8
    assert first.reproducibility_fingerprint == second.reproducibility_fingerprint
    assert any(effect.interpretation == "positive effect" for effect in first.effects)


def test_nonfinite_measurement_is_rejected():
    design = ExperimentDesigner().design(
        "test",
        [ExperimentVariable("x", (0, 1))],
        repetitions=1,
    )
    engine = ScientificExperimentEngine(bootstrap_samples=200)
    try:
        engine.run(design, lambda params, seed: {"score": float("nan")})
    except ValueError as exc:
        assert "not finite" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_replication_checker_flags_agreement():
    designer = ExperimentDesigner()
    design = designer.design("test", [ExperimentVariable("x", (0, 1))], repetitions=3, seed=1)

    def run(params, seed):
        return {"score": float(params["x"])}

    engine = ScientificExperimentEngine(bootstrap_samples=200)
    first = engine.run(design, run)
    second = engine.run(design, run)
    ok, detail = ReplicationChecker().compare(first, second)
    assert ok
    assert "agreement" in detail
