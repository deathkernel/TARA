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

    # out = max(0, 2*(-3)+4) = 0, so ReLU blocks the gradient.
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


def test_chain_rule():
    x = Value(3.0)
    y = x * x
    out = y + x
    out.backward()
    # d(x^2 + x)/dx = 2x + 1 = 7 at x=3
    assert x.grad == 7.0
