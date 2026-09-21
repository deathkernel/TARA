from experiments.compare_trained_lm_profiles import compare_trained


def test_trained_profile_comparison_uses_same_budget_and_reports_metrics():
    results = compare_trained(steps=1)

    assert set(results) == {"tiny", "scaled"}
    assert results["tiny"]["steps"] == results["scaled"]["steps"] == 1

    for result in results.values():
        assert result["parameters"] > 0
        assert result["initial_train"]["tokens"] > 0
        assert result["final_train"]["tokens"] > 0
        assert result["final_validation"]["tokens"] > 0
        assert result["final_train"]["loss"] >= 0.0
        assert result["final_validation"]["loss"] >= 0.0
        assert 0.0 <= result["final_train"]["accuracy"] <= 1.0
        assert result["final_train"]["entropy"] >= 0.0

    assert results["scaled"]["parameters"] > results["tiny"]["parameters"]
