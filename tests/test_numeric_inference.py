import math

from src.language_model import TinyLanguageModel
from src.numeric_inference import forward_numeric


def test_numeric_inference_matches_autograd_forward():
    model = TinyLanguageModel(
        vocab_size=7,
        embedding_dim=4,
        ff_dim=8,
        num_heads=2,
        num_layers=1,
        seed=7,
    )
    token_ids = [1, 2, 3]
    graph_logits = model.forward(token_ids)
    numeric_logits = model.forward_numeric(token_ids)

    assert len(graph_logits) == len(numeric_logits)
    for graph_row, numeric_row in zip(graph_logits, numeric_logits):
        for graph_value, numeric_value in zip(graph_row, numeric_row):
            assert math.isclose(graph_value.data, numeric_value, rel_tol=1e-10, abs_tol=1e-10)


def test_numeric_inference_does_not_modify_gradients():
    model = TinyLanguageModel(vocab_size=5, embedding_dim=4, ff_dim=8, num_heads=2, seed=3)
    model.forward([1, 2])[0][0].backward()
    before = [parameter.grad for parameter in model.parameters()]
    model.forward_numeric([1, 2, 3])
    assert before == [parameter.grad for parameter in model.parameters()]


def test_numeric_inference_rejects_empty_context():
    model = TinyLanguageModel(vocab_size=5, embedding_dim=4, ff_dim=8, num_heads=2, seed=3)
    try:
        model.forward_numeric([])
    except ValueError as exc:
        assert "token_ids" in str(exc)
    else:
        raise AssertionError("empty context should fail")
