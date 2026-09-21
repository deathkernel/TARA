import math

from experiments.trained_profile_comparison import compare_trained, train_profile
from experiments.compare_lm_profiles import TINY


def test_trained_profile_records_one_entry_per_step():
    result = train_profile(TINY, steps=2)
    assert len(result["history"]) == 2
    for entry in result["history"]:
        assert entry["step"] > 0
        assert math.isfinite(entry["batch_loss"])
        assert entry["gradient_norm"] >= 0.0
        assert entry["learning_rate"] > 0.0


def test_matched_training_comparison_reports_finite_metrics():
    results = compare_trained(steps=2)
    assert set(results) == {"tiny", "scaled"}
    for result in results.values():
        assert result["history"]
        assert math.isfinite(result["final_train"]["loss"])
        assert math.isfinite(result["final_validation"]["loss"])
        assert 0.0 <= result["final_train"]["accuracy"] <= 1.0
        assert result["final_train"]["entropy"] >= 0.0
