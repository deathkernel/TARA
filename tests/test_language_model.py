import math

from src.autograd import Value
from src.language_model import TinyLanguageModel, cross_entropy


def test_cross_entropy_is_finite_and_differentiable():
    logits = [Value(0.2), Value(-0.1), Value(0.7)]
    loss = cross_entropy(logits, 2)
    loss.backward()
    assert math.isfinite(loss.data)
    assert any(value.grad != 0.0 for value in logits)


def test_language_model_shapes_and_parameters():
    model = TinyLanguageModel(vocab_size=5, embedding_dim=4, ff_dim=8, seed=3)
    logits = model.forward([0, 1, 2])
    assert len(logits) == 3
    assert all(len(row) == 5 for row in logits)
    assert len(model.parameters()) > 0


def test_language_model_loss_backpropagates():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=5)
    model.zero_grad()
    loss = model.loss([0, 1, 2], [1, 2, 3])
    loss.backward()
    assert math.isfinite(loss.data)
    assert any(parameter.grad != 0.0 for parameter in model.parameters())


def test_next_token_is_valid():
    model = TinyLanguageModel(vocab_size=6, embedding_dim=3, ff_dim=6, seed=9)
    token = model.next_token([0, 1])
    assert 0 <= token < 6
