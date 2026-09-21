from experiments.compare_optimizers import SGD, compare
from src.autograd import Value


def test_sgd_moves_parameter_against_gradient():
    parameter = Value(2.0)
    parameter.grad = 3.0
    optimizer = SGD([parameter], learning_rate=0.1)
    norm = optimizer.step()
    assert parameter.data == 1.7
    assert norm == 3.0
    assert optimizer.step_count == 1


def test_matched_optimizer_comparison_has_same_budget():
    report = compare(steps=1)
    assert set(report) == {"SGD", "AdamW"}
    for result in report.values():
        assert result["steps"] == 1
        assert len(result["history"]) == 1
        assert result["initial_train"]["tokens"] > 0
        assert result["final_validation"]["tokens"] > 0


def test_optimizer_comparison_is_finite():
    report = compare(steps=2)
    for result in report.values():
        assert result["final_train"]["loss"] == result["final_train"]["loss"]
        assert result["final_validation"]["loss"] == result["final_validation"]["loss"]
