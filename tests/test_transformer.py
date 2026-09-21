import math
import random

import pytest

from src.autograd import Value
from src.positional import add_sinusoidal_position
from src.transformer import LayerNorm, Linear, MultiHeadCausalSelfAttention, TransformerBlock


def test_positional_encoding_changes_positions():
    sequence = [[Value(0.0), Value(0.0)], [Value(0.0), Value(0.0)]]
    encoded = add_sinusoidal_position(sequence)
    assert encoded[0][0].data != encoded[1][0].data


def test_layer_norm_is_close_to_zero_mean_and_unit_variance():
    layer_norm = LayerNorm(4)
    values = [Value(1.0), Value(2.0), Value(3.0), Value(4.0)]
    output = layer_norm.forward(values)
    mean = sum(value.data for value in output) / 4.0
    variance = sum((value.data - mean) ** 2 for value in output) / 4.0
    assert abs(mean) < 1e-5
    assert abs(variance - 1.0) < 1e-4


def test_layer_norm_rejects_non_positive_eps():
    with pytest.raises(ValueError):
        LayerNorm(4, eps=0.0)
    with pytest.raises(ValueError):
        LayerNorm(4, eps=-1e-5)


def test_linear_rejects_wrong_input_dimension():
    linear = Linear(3, 2, random.Random(3))
    with pytest.raises(ValueError):
        linear.forward([Value(1.0), Value(2.0)])
    with pytest.raises(ValueError):
        linear.forward([Value(1.0), Value(2.0), Value(3.0), Value(4.0)])


def test_multi_head_attention_shape():
    attention = MultiHeadCausalSelfAttention(
        4,
        num_heads=2,
        rng=random.Random(3),
    )
    sequence = [
        [Value(0.1), Value(0.2), Value(0.3), Value(0.4)]
        for _ in range(3)
    ]
    outputs = attention.forward(sequence)
    assert len(outputs) == 3
    assert all(len(token) == 4 for token in outputs)


def test_attention_is_causal():
    attention = MultiHeadCausalSelfAttention(
        4,
        num_heads=2,
        rng=random.Random(3),
    )
    prefix = [
        [Value(0.1), Value(0.2), Value(0.3), Value(0.4)],
        [Value(0.4), Value(0.3), Value(0.2), Value(0.1)],
    ]
    future_a = [Value(0.2), Value(0.8), Value(-0.4), Value(0.7)]
    future_b = [Value(-2.0), Value(3.0), Value(1.5), Value(-4.0)]

    outputs_a = attention.forward(prefix + [future_a])
    outputs_b = attention.forward(prefix + [future_b])

    for left, right in zip(outputs_a[:2], outputs_b[:2]):
        assert all(abs(a.data - b.data) < 1e-12 for a, b in zip(left, right))


def test_transformer_shape_and_gradients():
    model = TransformerBlock(4, ff_dim=8, num_heads=2, seed=11)
    sequence = [
        [Value(0.2), Value(-0.1), Value(0.3), Value(0.5)],
        [Value(0.4), Value(0.3), Value(-0.2), Value(0.1)],
    ]
    outputs = model.forward(sequence)
    assert len(outputs) == 2
    assert all(len(token) == 4 for token in outputs)
    loss = sum(outputs[-1])
    loss.backward()
    assert any(parameter.grad != 0.0 for parameter in model.parameters())
    assert all(math.isfinite(parameter.data) for parameter in model.parameters())


def test_transformer_rejects_incompatible_heads():
    with pytest.raises(ValueError):
        TransformerBlock(3, ff_dim=6, num_heads=2, seed=11)


def test_transformer_rejects_non_positive_heads_and_ff_dim():
    with pytest.raises(ValueError):
        TransformerBlock(4, ff_dim=8, num_heads=0, seed=11)
    with pytest.raises(ValueError):
        TransformerBlock(4, ff_dim=0, num_heads=2, seed=11)
    with pytest.raises(ValueError):
        TransformerBlock(4, ff_dim=-2, num_heads=2, seed=11)


def test_transformer_has_trainable_parameters():
    model = TransformerBlock(4, ff_dim=8, num_heads=2, seed=11)
    assert len(model.parameters()) > 0
