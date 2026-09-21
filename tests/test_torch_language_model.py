import pytest

torch = pytest.importorskip("torch")

from src.torch_language_model import FastTinyLanguageModel


def test_fast_model_shapes_and_loss():
    model = FastTinyLanguageModel(
        vocab_size=11,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=8,
        seed=3,
    )
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    logits = model(inputs)
    assert logits.shape == (2, 4, 11)
    loss = model.loss(inputs, targets)
    assert torch.isfinite(loss)


def test_fast_model_backpropagates():
    model = FastTinyLanguageModel(
        vocab_size=9,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=8,
        seed=5,
    )
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])
    loss = model.loss(inputs, targets)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )


def test_fast_attention_is_causal():
    torch.manual_seed(7)
    model = FastTinyLanguageModel(
        vocab_size=9,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=8,
        seed=5,
    )
    prefix = torch.tensor([[0, 1, 2]])
    first = model(prefix).detach()

    with_future_a = torch.tensor([[0, 1, 2, 3]])
    with_future_b = torch.tensor([[0, 1, 2, 4]])
    output_a = model(with_future_a).detach()
    output_b = model(with_future_b).detach()

    assert torch.allclose(output_a[:, :3], output_b[:, :3], atol=1e-6)
    assert torch.allclose(first, output_a[:, :3], atol=1e-6)


def test_fast_model_rejects_long_context():
    model = FastTinyLanguageModel(
        vocab_size=9,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=4,
        seed=5,
    )
    with pytest.raises(ValueError):
        model(torch.zeros((1, 5), dtype=torch.long))


def test_fast_model_rejects_empty_sequences():
    model = FastTinyLanguageModel(vocab_size=9, embedding_dim=8, ff_dim=16, num_heads=2)
    empty = torch.empty((1, 0), dtype=torch.long)
    with pytest.raises(ValueError):
        model(empty)
    with pytest.raises(ValueError):
        model.next_token(empty)
    with pytest.raises(ValueError):
        model.loss(empty, empty)


def test_fast_model_validates_constructor_arguments():
    with pytest.raises(ValueError):
        FastTinyLanguageModel(vocab_size=9, embedding_dim=8, ff_dim=16, num_heads=0)
    with pytest.raises(ValueError):
        FastTinyLanguageModel(vocab_size=9, embedding_dim=8, ff_dim=0, num_heads=2)
    with pytest.raises(ValueError):
        FastTinyLanguageModel(vocab_size=9, embedding_dim=8, ff_dim=16, num_heads=2, max_context=0)


def test_fast_model_validates_loss_targets():
    model = FastTinyLanguageModel(vocab_size=9, embedding_dim=8, ff_dim=16, num_heads=2)
    inputs = torch.tensor([[0, 1, 2]])
    with pytest.raises(ValueError):
        model.loss(inputs, torch.tensor([[1, 2]]))
    with pytest.raises(ValueError):
        model.loss(inputs, torch.tensor([[1, 2, 9]]))
    with pytest.raises(ValueError):
        model.loss(inputs, torch.tensor([[1, 2, -1]]))


def test_fast_model_constructor_does_not_change_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(5)

    torch.manual_seed(123)
    FastTinyLanguageModel(
        vocab_size=9,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=8,
        seed=5,
    )
    actual = torch.rand(5)

    assert torch.equal(actual, expected)
