import math

from src.tara_mind.cognition.scientific_reasoning import (
    BayesianReasoner,
    CausalGraph,
    Evidence,
    Hypothesis,
    Measurement,
    weighted_measurement_mean,
)


def test_bayesian_evidence_updates_posterior():
    reasoner = BayesianReasoner()
    hypothesis = Hypothesis("H", 0.5)
    posterior = reasoner.posterior(hypothesis, [Evidence(0.9, 0.1)])
    assert math.isclose(posterior, 0.9, rel_tol=1e-9)


def test_causal_intervention_propagates_effects():
    graph = CausalGraph()
    graph.add_edge("A", "B", 2.0)
    graph.add_edge("B", "C", 3.0)
    result = graph.predict_intervention({"A": 1.0})
    assert result["B"] == 2.0
    assert result["C"] == 6.0


def test_measurements_use_inverse_variance_weighting():
    result = weighted_measurement_mean([
        Measurement(10.0, 1.0, "m"),
        Measurement(12.0, 2.0, "m"),
    ])
    assert math.isclose(result.value, 10.4, rel_tol=1e-9)
    assert result.uncertainty < 1.0
