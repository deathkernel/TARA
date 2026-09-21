import math

import pytest

from src.autograd import Value
from src.optimizers import AdamW


def test_adamw_moves_parameter_against_gradient():
    parameter = Value(1.0)
    parameter.grad = 2.0
    optimizer = AdamW([parameter], learning_rate=0.1)
    norm = optimizer.step()
    assert math.isclose(norm, 2.0)
    assert parameter.data < 1.0


def test_adamw_bias_correction_makes_first_update_stable():
    parameter = Value(1.0)
    parameter.grad = 1.0
    optimizer = AdamW([parameter], learning_rate=0.1)
    optimizer.step()
    assert math.isclose(parameter.data, 0.9, rel_tol=1e-6, abs_tol=1e-6)


def test_adamw_decoupled_weight_decay():
    parameter = Value(2.0)
    parameter.grad = 0.0
    optimizer = AdamW([parameter], learning_rate=0.1, weight_decay=0.5)
    optimizer.step()
    assert math.isclose(parameter.data, 1.9, rel_tol=1e-9)


def test_adamw_state_round_trip():
    first = Value(1.0)
    first.grad = 0.5
    optimizer = AdamW([first], learning_rate=0.01, weight_decay=0.01)
    optimizer.step()
    state = optimizer.state_dict()

    second = Value(first.data)
    restored = AdamW([second], learning_rate=0.2)
    restored.load_state_dict(state)
    assert restored.step_count == 1
    assert restored.learning_rate == 0.01
    assert restored.weight_decay == 0.01
    assert restored.state_dict() == state


def test_adamw_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        AdamW([Value(1.0)], learning_rate=0.0)
    with pytest.raises(ValueError):
        AdamW([Value(1.0)], betas=(1.0, 0.999))
    with pytest.raises(ValueError):
        AdamW([Value(1.0)], epsilon=0.0)
    with pytest.raises(ValueError):
        AdamW([Value(1.0)], weight_decay=-1.0)


def test_adamw_failed_step_does_not_advance_state():
    parameter = Value(1.0)
    parameter.grad = 0.0
    optimizer = AdamW([parameter])
    before = optimizer.state_dict()
    parameter.grad = float("nan")
    with pytest.raises(ValueError):
        optimizer.step()
    assert optimizer.state_dict() == before
    assert parameter.data == 1.0


def test_adamw_rejects_invalid_state():
    parameter = Value(1.0)
    optimizer = AdamW([parameter])
    state = optimizer.state_dict()
    state.pop("beta1")
    with pytest.raises(ValueError):
        optimizer.load_state_dict(state)
