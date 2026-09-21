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
