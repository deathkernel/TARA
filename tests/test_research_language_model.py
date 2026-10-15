import pytest

torch = pytest.importorskip("torch")

from src.research_language_model import (
    GroupedQueryAttention,
    ResearchTinyLanguageModel,
    RotaryEmbedding,
    SwiGLU,
    count_parameters,
)


def make_model(**overrides):
    options = {
        "vocab_size": 19,
        "embedding_dim": 16,
        "ff_dim": 32,
        "num_heads": 4,
        "num_kv_heads": 2,
        "max_context": 8,
        "num_layers": 2,
        "seed": 13,
    }
    options.update(overrides)
    return ResearchTinyLanguageModel(**options)


def test_research_model_shapes_loss_and_parameter_count():
    model = make_model()
    inputs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

    logits = model(inputs)
    loss = model.loss(inputs, targets)

    assert logits.shape == (2, 4, 19)
    assert torch.isfinite(loss)
    assert count_parameters(model) > 0


def test_research_model_backpropagates():
    model = make_model()
    inputs = torch.tensor([[0, 1, 2, 3]])
    targets = torch.tensor([[1, 2, 3, 4]])

    model.loss(inputs, targets).backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_attention_is_causal():
    model = make_model()
    model.eval()
    prefix = torch.tensor([[0, 1, 2]])
    with_future_a = torch.tensor([[0, 1, 2, 3]])
    with_future_b = torch.tensor([[0, 1, 2, 4]])

    prefix_logits = model(prefix)
    logits_a = model(with_future_a)
    logits_b = model(with_future_b)

    assert torch.allclose(prefix_logits, logits_a[:, :3], atol=1e-5)
    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5)


def test_grouped_query_attention_uses_fewer_kv_projection_parameters():
    grouped = GroupedQueryAttention(32, num_heads=4, num_kv_heads=2)
    full = GroupedQueryAttention(32, num_heads=4, num_kv_heads=4)
    assert grouped.key.out_features == 16
    assert grouped.value.out_features == 16
    assert full.key.out_features == 32
    assert count_parameters(grouped) < count_parameters(full)


def test_rotary_embedding_preserves_shape_and_finiteness():
    rope = RotaryEmbedding(head_dim=4)
    query = torch.randn(2, 4, 6, 4)
    key = torch.randn(2, 2, 6, 4)
    rotated_query, rotated_key = rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape
    assert torch.isfinite(rotated_query).all()
    assert torch.isfinite(rotated_key).all()


def test_swiglu_shape():
    layer = SwiGLU(dimension=16, ff_dimension=24)
    assert layer(torch.randn(2, 5, 16)).shape == (2, 5, 16)


def test_generation_respects_requested_length_and_top_k():
    model = make_model()
    model.eval()
    prompt = torch.tensor([[1, 2, 3]])
    generator = torch.Generator().manual_seed(5)

    output = model.generate(prompt, max_new_tokens=4, top_k=5, generator=generator)

    assert output.shape == (1, 7)
    assert torch.equal(output[:, :3], prompt)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 1},
        {"embedding_dim": 15, "num_heads": 4},
        {"embedding_dim": 12, "num_heads": 4},  # odd RoPE head dimension
        {"num_heads": 4, "num_kv_heads": 3},
        {"num_layers": 0},
        {"max_context": 0},
        {"dropout": 1.0},
    ],
)
def test_research_model_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        make_model(**kwargs)


def test_research_model_rejects_empty_and_overlong_sequences():
    model = make_model(max_context=4)
    with pytest.raises(ValueError):
        model(torch.empty((1, 0), dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.zeros((1, 5), dtype=torch.long))


def test_constructor_does_not_change_global_rng():
    torch.manual_seed(123)
    expected = torch.rand(5)
    torch.manual_seed(123)

    make_model(seed=9)
    actual = torch.rand(5)

    assert torch.equal(actual, expected)
