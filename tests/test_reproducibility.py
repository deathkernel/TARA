import random

from src.evaluation import build_report, compare_metric
from src.generation import sample_from_logits


def test_sampling_is_reproducible_with_explicit_rng_seed():
    logits = [0.2, 1.1, 0.7, 2.0]
    first = sample_from_logits(logits, temperature=0.9, top_k=3, rng=random.Random(17))
    second = sample_from_logits(logits, temperature=0.9, top_k=3, rng=random.Random(17))
    assert first == second


def test_evaluation_report_is_deterministic_for_same_results():
    results = [
        compare_metric("loss", 0.25, maximum=0.5),
        compare_metric("accuracy", 0.75, minimum=0.7),
    ]
    first = build_report(results, {"seed": 17, "context_length": 8})
    second = build_report(results, {"seed": 17, "context_length": 8})
    assert first == second
