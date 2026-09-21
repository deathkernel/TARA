import math

from experiments.train_scaled_lm import train


def test_scaled_training_reduces_training_loss():
    result = train(steps=4)
    assert result["final_train"]["loss"] <= result["initial_train"]["loss"]
    assert math.isfinite(result["final_validation"]["loss"])
    assert 0.0 <= result["final_train"]["accuracy"] <= 1.0
    assert result["final_train"]["entropy"] >= 0.0


def test_scaled_training_records_finite_step_history_and_optimizer_state():
    result = train(steps=3)
    assert len(result["history"]) == 3
    for item in result["history"]:
        assert item["step"] >= 0
        assert math.isfinite(item["loss"])
        assert item["learning_rate"] > 0.0
        assert math.isfinite(item["gradient_norm"])
    assert result["config"]["optimizer"] == "AdamW"
    assert result["optimizer"]["step_count"] == 3
