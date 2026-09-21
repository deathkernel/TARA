from experiments.compare_lm_profiles import build_profile, compare, parameter_count


def test_scaled_profile_has_more_parameters_than_tiny_profile():
    # This test checks capacity only. Avoid running the full language evaluation
    # just to count parameters; the evaluation test below covers the metrics.
    tiny_model, _, _ = build_profile(
        {"embedding_dim": 8, "ff_dim": 16, "num_heads": 2, "num_layers": 1}
    )
    scaled_model, _, _ = build_profile(
        {"embedding_dim": 32, "ff_dim": 64, "num_heads": 4, "num_layers": 2}
    )
    assert parameter_count(scaled_model) > parameter_count(tiny_model)


def test_profile_comparison_reports_finite_metrics():
    results = compare()
    for result in results.values():
        assert result["parameters"] > 0
        for split in ("train", "validation"):
            assert result[split]["tokens"] > 0
            assert result[split]["loss"] >= 0.0
            assert 0.0 <= result[split]["accuracy"] <= 1.0
            assert result[split]["entropy"] >= 0.0
