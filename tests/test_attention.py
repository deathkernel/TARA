import math

from src.attention import SelfAttention, _softmax
from src.autograd import Value


def test_softmax_sums_to_one():
    weights = _softmax([Value(1.0), Value(2.0), Value(3.0)])
    assert math.isclose(sum(weight.data for weight in weights), 1.0, rel_tol=1e-9)
    assert all(weight.data > 0 for weight in weights)


def test_causal_attention_does_not_read_future_tokens():
    attention = SelfAttention(1)
    sequence = [[Value(1.0)], [Value(2.0)], [Value(100.0)]]
    outputs = attention.forward(sequence)
    assert math.isclose(outputs[0][0].data, 1.0, rel_tol=1e-9)
    assert outputs[1][0].data < 2.0
    assert outputs[2][0].data > outputs[1][0].data


def test_attention_is_differentiable():
    first = Value(1.0)
    second = Value(2.0)
    outputs = SelfAttention(1).forward([[first], [second]])
    loss = outputs[1][0]
    loss.backward()
    assert first.grad != 0.0
    assert second.grad != 0.0
