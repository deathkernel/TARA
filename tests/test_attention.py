import math

from src.attention import SelfAttention, _softmax
from src.autograd import Value


def test_softmax_sums_to_one():
    weights = _softmax([Value(1.0), Value(2.0), Value(3.0)])
    assert math.isclose(sum(weight.data for weight in weights), 1.0, rel_tol=1e-9)
    assert all(weight.data > 0 for weight in weights)


def test_causal_attention_does_not_read_future_tokens():
    attention = SelfAttention(2, seed=7)
    prefix = [[Value(1.0), Value(0.5)], [Value(2.0), Value(-1.0)]]
    with_future = prefix + [[Value(100.0), Value(50.0)]]
    first_without = attention.forward(prefix)[0]
    first_with = attention.forward(with_future)[0]
    assert all(math.isclose(a.data, b.data, rel_tol=1e-9, abs_tol=1e-9)
               for a, b in zip(first_without, first_with))


def test_attention_is_differentiable_and_has_parameters():
    first = Value(1.0)
    second = Value(2.0)
    model = SelfAttention(1, seed=3)
    outputs = model.forward([[first], [second]])
    loss = outputs[1][0]
    loss.backward()
    assert first.grad != 0.0
    assert second.grad != 0.0
    assert len(model.parameters()) == 6
