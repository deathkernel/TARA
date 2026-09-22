import pytest

torch = pytest.importorskip("torch")

from src.torch_language_model import FastTinyLanguageModel


def test_deeper_model_has_requested_depth_and_forward_shape():
    model = FastTinyLanguageModel(
        vocab_size=17,
        embedding_dim=16,
        ff_dim=32,
        num_heads=4,
        max_context=12,
        num_layers=3,
        dropout=0.1,
        tie_embeddings=True,
        seed=7,
    )
    tokens = torch.randint(0, 17, (2, 8))
    logits = model(tokens)
    assert logits.shape == (2, 8, 17)
    assert len(model.transformer) == 3
    assert model.lm_head.weight is model.embedding.weight


def test_dropout_is_disabled_in_eval_mode():
    model = FastTinyLanguageModel(
        vocab_size=11,
        embedding_dim=12,
        ff_dim=24,
        num_heads=3,
        max_context=8,
        num_layers=2,
        dropout=0.5,
        seed=3,
    )
    tokens = torch.randint(0, 11, (1, 6))
    model.eval()
    first = model(tokens)
    second = model(tokens)
    assert torch.equal(first, second)


def test_model_rejects_invalid_depth_and_dropout():
    with pytest.raises(ValueError, match="num_layers"):
        FastTinyLanguageModel(8, num_layers=0)
    with pytest.raises(ValueError, match="dropout"):
        FastTinyLanguageModel(8, dropout=1.0)
