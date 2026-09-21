import math

from src.language_benchmarks import evaluate
from src.language_dataset import CausalTextDataset
from src.language_model import TinyLanguageModel


def test_language_benchmark_reports_core_metrics():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    dataset = CausalTextDataset([0, 1, 2, 3], context_length=2)

    metrics = evaluate(model, dataset)

    assert set(metrics) == {"loss", "perplexity", "accuracy", "tokens"}
    assert metrics["tokens"] == 3
    assert metrics["loss"] > 0.0
    assert math.isclose(metrics["perplexity"], math.exp(metrics["loss"]))
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_language_benchmark_does_not_change_parameters():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    dataset = CausalTextDataset([0, 1, 2, 3], context_length=2)
    before = [parameter.data for parameter in model.parameters()]

    evaluate(model, dataset)

    assert [parameter.data for parameter in model.parameters()] == before
