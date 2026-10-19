from __future__ import annotations

from src.intelligence_benchmark import BenchmarkCase, normalized_text_score
from src.model_capability_runner import ModelCapabilityRunner


class FakeTokenizer:
    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeModel:
    def forward_numeric(self, token_ids):
        logits = [0.0] * 128
        logits[ord("!")] = 1.0
        return [logits]


def test_runner_is_deterministic_and_returns_generated_suffix():
    runner = ModelCapabilityRunner(FakeModel(), FakeTokenizer(), max_new_tokens=2)
    assert runner.generate("x") == "!!"
    cases = (BenchmarkCase("echo", "coding", "x", "!!", normalized_text_score),)
    first = runner.evaluate(cases)
    second = runner.evaluate(cases)
    assert first.benchmark.fingerprint == second.benchmark.fingerprint
    assert first.fingerprint == second.fingerprint
    assert first.benchmark.overall_score == 1.0
    assert first.benchmark.results[0].error is None


class SequencedFakeModel:
    """Return a different token per generation to test repeated prompts."""

    def __init__(self):
        self.calls = 0

    def forward_numeric(self, token_ids):
        token = ord("!") if self.calls == 0 else ord("?")
        self.calls += 1
        logits = [0.0] * 128
        logits[token] = 1.0
        return [logits]


def test_runner_keeps_duplicate_prompts_associated_with_their_cases():
    runner = ModelCapabilityRunner(SequencedFakeModel(), FakeTokenizer(), max_new_tokens=1)
    cases = (
        BenchmarkCase("first", "coding", "same prompt", "!", normalized_text_score),
        BenchmarkCase("second", "coding", "same prompt", "?", normalized_text_score),
    )

    result = runner.evaluate(cases)

    assert result.observations == {"first": "!", "second": "?"}
    assert result.benchmark.overall_score == 1.0
    assert [case.error for case in result.benchmark.results] == [None, None]
