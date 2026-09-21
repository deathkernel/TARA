from experiments.compare_lm_profiles import TINY, SCALED, VOCAB_SIZE, parameter_count
from src.language_model import TinyLanguageModel


def test_scaled_profile_has_more_parameters_than_tiny_profile():
    # Capacity-only regression: construct the models directly. Building the
    # tokenizer and causal datasets is unrelated to parameter-count behavior
    # and made this test unnecessarily expensive on the scalar autodiff core.
    tiny_model = TinyLanguageModel(VOCAB_SIZE, seed=7, **TINY)
    scaled_model = TinyLanguageModel(VOCAB_SIZE, seed=7, **SCALED)
    assert parameter_count(scaled_model) > parameter_count(tiny_model)


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
