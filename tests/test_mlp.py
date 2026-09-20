"""Tests for TARA's autograd-native MLP."""

from src.autograd import Value
from src.mlp import MLP


def test_mlp_parameter_count():
    model = MLP(2, [3, 3, 1], seed=7)
    # 3*(2 weights + bias) + 3*(3 weights + bias) + 1*(3 weights + bias)
    assert len(model.parameters()) == 25


def test_mlp_output_is_value_for_single_output():
    model = MLP(2, [3, 1], seed=7)
    output = model.forward([1.0, -1.0])
    assert isinstance(output, Value)


def test_mlp_parameters_are_values():
    model = MLP(2, [2, 1], seed=7)
    assert all(isinstance(parameter, Value) for parameter in model.parameters())


def test_mlp_backward_reaches_parameters():
    model = MLP(2, [2, 1], seed=7)
    output = model.forward([1.0, 2.0])
    loss = output ** 2
    loss.backward()
    assert any(parameter.grad != 0.0 for parameter in model.parameters())
