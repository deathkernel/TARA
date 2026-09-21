import pytest

from src.autograd import Value
from src.gradcheck import check_parameter_gradients, numerical_gradient
from src.losses import MSELoss, mean
from src.mlp import MLP


def test_mlp_gradients_match_centered_finite_difference():
    model = MLP(2, [3, 2, 1], seed=11)
    inputs = [[0.2, -0.4], [0.7, 0.3]]
    targets = [0.6, -0.2]
    loss = MSELoss()

    model.zero_grad()
    objective = mean(loss(model.forward(x), y) for x, y in zip(inputs, targets))
    objective.backward()

    diagnostics = check_parameter_gradients(
        lambda: mean(loss(model.forward(x), y) for x, y in zip(inputs, targets)).data,
        model.parameters(),
        epsilon=1e-6,
        tolerance=1e-5,
    )

    assert len(diagnostics) == len(model.parameters())
    assert max(item["relative_error"] for item in diagnostics) < 1e-5


def test_numerical_gradient_restores_parameter_after_failure():
    parameter = Value(2.5)
    original = parameter.data

    def failing_loss():
        raise RuntimeError("intentional test failure")

    with pytest.raises(RuntimeError):
        numerical_gradient(failing_loss, parameter)

    assert parameter.data == original


def test_numerical_gradient_rejects_non_positive_epsilon():
    parameter = Value(1.0)
    with pytest.raises(ValueError):
        numerical_gradient(lambda: parameter.data ** 2, parameter, epsilon=0)
