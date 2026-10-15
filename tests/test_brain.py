import math

import pytest

torch = pytest.importorskip("torch")

from src.brain import TARABrain
from src.language_model import TinyLanguageModel
from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def make_brain():
    text = "tara can reason with context and memory."
    tokenizer = CharTokenizer(text)
    model = TinyLanguageModel(tokenizer.vocab_size, seed=5)
    return TARABrain(model, tokenizer, seed=5)


def test_brain_connects_observation_memory_and_generation():
    brain = make_brain()
    response = brain.respond("tara", remember_key="prompt", max_new_tokens=3)
    assert response.text.startswith("tara")
    assert response.observations[-1] == "tara"
    assert response.recalled


def test_brain_supports_goal_planning_and_verification():
    brain = make_brain()
    brain.set_goal("finish task", ("done",))
    brain.plan(["inspect", "finish"])
    assert brain.agent.next_task().task == "inspect"
    brain.verify_result("ok", "ok")
    assert brain.agent.next_task().task == "finish"


def test_brain_top_p_generation_is_finite_and_bounded():
    brain = make_brain()
    output = brain.generate("tara", max_new_tokens=5, top_p=0.8, temperature=1.0)
    assert output.startswith("tara")
    assert len(output) >= len("tara")
    assert all(math.isfinite(value) for value in brain.model.forward_numeric(brain.tokenizer.encode("tara"))[-1])


def test_brain_rejects_invalid_generation_arguments():
    brain = make_brain()
    for kwargs in ({"max_new_tokens": 0}, {"temperature": 0}, {"top_p": 0}, {"top_p": 1.1}):
        try:
            brain.generate("tara", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation argument should fail")


def test_reasoning_context_is_bounded_without_truncating_user_request(monkeypatch):
    brain = make_brain()
    monkeypatch.setenv("TARA_MAX_CONTEXT_CHARS", "100")

    context = brain.build_reasoning_context(
        "important user request",
        recalled=(),
    )

    assert len(context) <= 100
    assert "important user request" in context
    assert "trimmed for low-resource mode" in context

def test_brain_passes_emotional_guidance_to_language_provider():
    brain = make_brain()

    class FakeProvider:
        model = "test-model"

        def __init__(self):
            self.system = ""

        def complete(self, prompt, *, system=None, max_completion_tokens=32, temperature=0.6):
            self.system = system or ""
            return "I'm sorry you're feeling sad. Want to tell me what happened?"

    provider = FakeProvider()
    brain.llm_provider = provider
    response = brain.respond("I feel sad", max_new_tokens=32)

    assert response.emotion == "sadness"
    assert brain.last_emotion == "sadness"
    assert "Response style:" in provider.system
    assert "never claim TARA literally feels emotions" in provider.system
