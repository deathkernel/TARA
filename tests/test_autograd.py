from math import isclose, tanh

import pytest

from src.autograd import Value


def test_addition_gradients():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_multiplication_gradients():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_composed_expression():
    x = Value(2.0)
    y = Value(-3.0)
    z = Value(4.0)
    out = (x * y + z).relu()
    out.backward()
    assert out.data == 0.0
    assert x.grad == 0.0
    assert y.grad == 0.0
    assert z.grad == 0.0


def test_power_and_division():
    x = Value(2.0)
    out = x**3
    out.backward()
    assert out.data == 8.0
    assert x.grad == 12.0


def test_zero_power_has_zero_gradient():
    x = Value(0.0)
    out = x**0
    out.backward()
    assert out.data == 1.0
    assert x.grad == 0.0


def test_zero_negative_power_is_rejected():
    with pytest.raises(ValueError, match="negative power"):
        Value(0.0) ** -1


def test_negative_base_fractional_power_is_rejected():
    with pytest.raises(ValueError, match="integer exponent"):
        Value(-2.0) ** 0.5


def test_chain_rule():
    x = Value(3.0)
    y = x * x
    out = y + x
    out.backward()
    assert x.grad == 7.0


def test_tanh_forward_and_gradient():
    x = Value(0.7)
    out = x.tanh()
    out.backward()
    expected = tanh(0.7)
    assert isclose(out.data, expected, rel_tol=1e-12, abs_tol=1e-12)
    assert isclose(x.grad, 1.0 - expected**2, rel_tol=1e-12, abs_tol=1e-12)
