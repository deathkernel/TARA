import random

import pytest

from src.generation import generate, sample_from_logits
from src.tokenizer import CharTokenizer


class FakeModel:
    def __init__(self):
        self.calls = []

    def forward(self, token_ids):
        self.calls.append(list(token_ids))
        return [[0.0, 1.0, 2.0]]


def test_greedy_like_sampling_with_top_k_is_reproducible():
    logits = [0.0, 1.0, 3.0, 2.0]
    assert sample_from_logits(logits, top_k=1, rng=random.Random(1)) == 2
    a = sample_from_logits(logits, temperature=0.8, top_k=3, rng=random.Random(7))
    b = sample_from_logits(logits, temperature=0.8, top_k=3, rng=random.Random(7))
    assert a == b


def test_top_p_keeps_high_probability_prefix():
    assert sample_from_logits([5.0, 4.0, 0.0], top_p=0.5, rng=random.Random(2)) == 0


def test_sampling_rejects_invalid_controls():
    with pytest.raises(ValueError):
        sample_from_logits([1.0], temperature=0.0)
    with pytest.raises(ValueError):
        sample_from_logits([1.0], top_k=0)
    with pytest.raises(ValueError):
        sample_from_logits([1.0], top_p=0.0)


def test_generate_respects_context_and_length():
    tokenizer = CharTokenizer("abc")
    model = FakeModel()
    output = generate(
        model, tokenizer, "a", length=3, context_length=2,
        temperature=1.0, top_k=1, rng=random.Random(1)
    )
    assert len(output) == 4
    assert model.calls == [[1], [1, 2], [2, 2]]


def test_generate_accepts_numeric_model_logits():
    tokenizer = CharTokenizer("abc")
    model = FakeModel()
    assert generate(model, tokenizer, "a", length=1, top_k=1) == "ac"
