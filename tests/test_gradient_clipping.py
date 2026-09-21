import math

import pytest

from src.autograd import Value
from src.gradient_clipping import clip_grad_norm_


def test_gradient_norm_is_unchanged_below_threshold():
    parameters = [Value(0.0), Value(0.0)]
    parameters[0].grad = 3.0
    parameters[1].grad = 4.0

    norm = clip_grad_norm_(parameters, max_norm=5.0)

    assert math.isclose(norm, 5.0)
    assert parameters[0].grad == 3.0
    assert parameters[1].grad == 4.0


def test_large_gradient_norm_is_scaled_to_threshold():
    parameters = [Value(0.0), Value(0.0)]
    parameters[0].grad = 6.0
    parameters[1].grad = 8.0

    norm = clip_grad_norm_(parameters, max_norm=5.0)

    assert math.isclose(norm, 10.0)
    clipped_norm = math.sqrt(sum(parameter.grad ** 2 for parameter in parameters))
    assert math.isclose(clipped_norm, 5.0, rel_tol=1e-9)
    assert math.isclose(parameters[0].grad, 3.0)
    assert math.isclose(parameters[1].grad, 4.0)


def test_empty_parameters_have_zero_norm():
    assert clip_grad_norm_([], max_norm=1.0) == 0.0


def test_invalid_max_norm_is_rejected():
    with pytest.raises(ValueError):
        clip_grad_norm_([Value(1.0)], max_norm=0.0)


def test_non_finite_gradient_is_rejected():
    parameter = Value(1.0)
    parameter.grad = float("inf")
    with pytest.raises(ValueError):
        clip_grad_norm_([parameter], max_norm=1.0)
