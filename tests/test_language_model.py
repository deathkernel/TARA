import math
import random

from src.autograd import Value
from src.language_model import SmallLanguageModel, TinyLanguageModel, cross_entropy


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


def test_sample_next_token_is_valid_and_reproducible():
    model = TinyLanguageModel(vocab_size=6, embedding_dim=3, ff_dim=6, seed=9)
    token_a = model.sample_next_token([0, 1], temperature=0.8, top_k=3, rng=random.Random(42))
    token_b = model.sample_next_token([0, 1], temperature=0.8, top_k=3, rng=random.Random(42))
    assert token_a == token_b
    assert 0 <= token_a < 6


def test_sample_next_token_rejects_invalid_temperature_and_top_k():
    model = TinyLanguageModel(vocab_size=6, embedding_dim=3, ff_dim=6, seed=9)
    try:
        model.sample_next_token([0, 1], temperature=0.0)
        assert False
    except ValueError:
        pass
    try:
        model.sample_next_token([0, 1], top_k=0)
        assert False
    except ValueError:
        pass


def test_small_language_model_has_explicit_pc_friendly_capacity_profile():
    model = SmallLanguageModel(vocab_size=16, seed=11)
    assert model.transformer.num_layers == 2
    assert len(model.embedding.parameters()) == 16 * 32
    assert len(model.forward([0, 1, 2])) == 3
    assert all(len(row) == 16 for row in model.forward([0, 1, 2]))
    assert len(model.parameters()) > len(TinyLanguageModel(vocab_size=16, seed=11).parameters())
