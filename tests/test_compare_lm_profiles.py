from experiments.compare_lm_profiles import compare, parameter_count


def test_scaled_profile_has_more_parameters_than_tiny_profile():
    results = compare()
    assert results["scaled"]["parameters"] > results["tiny"]["parameters"]


def test_profile_comparison_reports_finite_metrics():
    results = compare()
    for result in results.values():
        assert result["parameters"] > 0
        for split in ("train", "validation"):
            assert result[split]["tokens"] > 0
            assert result[split]["loss"] >= 0.0
            assert 0.0 <= result[split]["accuracy"] <= 1.0
            assert result[split]["entropy"] >= 0.0
