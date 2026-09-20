import math

from src.autograd import Value
from src.positional import add_sinusoidal_position
from src.transformer import TransformerBlock


def test_positional_encoding_changes_positions():
    sequence = [[Value(0.0), Value(0.0)], [Value(0.0), Value(0.0)]]
    encoded = add_sinusoidal_position(sequence)
    assert encoded[0][0].data != encoded[1][0].data


def test_transformer_shape_and_gradients():
    model = TransformerBlock(2, ff_dim=4, seed=11)
    sequence = [[Value(0.2), Value(-0.1)], [Value(0.4), Value(0.3)]]
    outputs = model.forward(sequence)
    assert len(outputs) == 2
    assert all(len(token) == 2 for token in outputs)
    loss = outputs[-1][0] + outputs[-1][1]
    loss.backward()
    assert any(parameter.grad != 0.0 for parameter in model.parameters())
    assert all(math.isfinite(parameter.data) for parameter in model.parameters())


def test_transformer_has_trainable_parameters():
    model = TransformerBlock(2, ff_dim=4, seed=11)
    assert len(model.parameters()) > 0
