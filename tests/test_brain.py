from types import SimpleNamespace

import pytest

from src.__main__ import chat_loop
from src.brain import TARABrain
from src.memory import LongTermMemory


class FakeProvider:
    default_max_tokens = 64

    def __init__(self, answers=None):
        self.calls = []
        self.answers = list(answers or ["test response"])

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.answers.pop(0) if self.answers else "test response"


def test_context_is_bounded_and_last_turn_is_included():
    provider = FakeProvider(["one", "two", "three"])
    brain = TARABrain(llm_provider=provider, max_history_turns=2)
    brain.respond("first")
    brain.respond("second")
    brain.respond("third")
    assert len(brain.conversation_history) == 4
    assert "User: first" not in provider.calls[2][0]
    assert "User: second" in provider.calls[2][0]
    assert "TARA: two" in provider.calls[2][0]


def test_empty_provider_response_does_not_corrupt_history():
    brain = TARABrain(llm_provider=FakeProvider([""]))
    with pytest.raises(RuntimeError, match="empty response"):
        brain.respond("hello")
    assert brain.conversation_history == []


def test_invalid_prompt_and_token_limit_are_rejected():
    brain = TARABrain(llm_provider=FakeProvider())
    with pytest.raises(ValueError, match="prompt"):
        brain.respond("  ")
    with pytest.raises(ValueError, match="positive integer"):
        brain.respond("hello", max_new_tokens=True)


def test_language_and_personality_are_in_system_prompt():
    provider = FakeProvider()
    brain = TARABrain(llm_provider=provider, language="Hindi", personality="professional")
    brain.respond("hello")
    system = provider.calls[0][1]["system"]
    assert "Respond in Hindi" in system
    assert "professional" in system.lower()


def test_clear_context_preserves_saved_memory():
    memory = LongTermMemory()
    memory.remember("language", "Hindi")
    brain = TARABrain(llm_provider=FakeProvider(["first", "second"]), memory=memory)
    brain.respond("hello")
    brain.clear_context()
    assert brain.conversation_history == []
    assert memory.recall("language") == "Hindi"


def test_cli_clear_command_does_not_call_provider_for_command():
    provider = FakeProvider(["old response", "new response"])
    brain = TARABrain(llm_provider=provider)
    inputs = iter(["hello", "/clear", "again", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert len(provider.calls) == 2
    assert "Active conversation context cleared" in "\n".join(output)
    assert "User: hello" not in provider.calls[1][0]


def test_cli_help_mentions_clear_command():
    brain = SimpleNamespace(respond=lambda *args, **kwargs: None, llm_provider=None)
    inputs = iter(["/help", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert "/clear" in "\n".join(output)
