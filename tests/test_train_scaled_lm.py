import math

from experiments.train_scaled_lm import train


def test_scaled_training_reduces_training_loss():
    result = train(steps=4)
    assert result["final_train"]["loss"] <= result["initial_train"]["loss"]
    assert math.isfinite(result["final_validation"]["loss"])
    assert 0.0 <= result["final_train"]["accuracy"] <= 1.0
    assert result["final_train"]["entropy"] >= 0.0
