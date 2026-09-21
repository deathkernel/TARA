from experiments.compare_lm_profiles import (
    TINY,
    SCALED,
    VOCAB_SIZE,
    parameter_count,
    profile_parameter_count,
)
from src.language_model import TinyLanguageModel


def test_scaled_profile_has_more_parameters_than_tiny_profile():
    # Capacity-only regression is analytical: constructing the scaled model
    # creates thousands of scalar autodiff objects for no reason.
    assert profile_parameter_count(SCALED) > profile_parameter_count(TINY)


def test_analytical_parameter_count_matches_tiny_model():
    # Validate the formula against a real model once, using the tiny profile.
    model = TinyLanguageModel(VOCAB_SIZE, seed=7, **TINY)
    assert profile_parameter_count(TINY) == parameter_count(model)


def test_profile_comparison_reports_finite_metrics():
    from experiments.compare_lm_profiles import compare

    results = compare()
    for result in results.values():
        assert result["parameters"] > 0
        for split in ("train", "validation"):
            assert result[split]["tokens"] > 0
            assert result[split]["loss"] >= 0.0
            assert 0.0 <= result[split]["accuracy"] <= 1.0
            assert result[split]["entropy"] >= 0.0
