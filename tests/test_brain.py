from threading import Event, Thread
from types import SimpleNamespace

import pytest

from src.__main__ import _positive_int_env, chat_loop
from src.brain import (
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    MAX_PROVIDER_RESPONSE_CHARS,
    MAX_USER_MESSAGE_CHARS,
    TARABrain,
)
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
    provider = FakeProvider(["one", "two", "three", "four"])
    brain = TARABrain(llm_provider=provider, max_history_turns=2)
    brain.respond("first")
    brain.respond("second")
    brain.respond("third")
    assert "User: first" in provider.calls[2][0]
    assert "User: second" in provider.calls[2][0]
    assert "TARA: two" in provider.calls[2][0]
    brain.respond("fourth")
    assert len(brain.conversation_history) == 4
    assert "User: first" not in provider.calls[3][0]
    assert "User: second" in provider.calls[3][0]
    assert "TARA: three" in provider.calls[3][0]


def test_empty_provider_response_does_not_corrupt_history():
    brain = TARABrain(llm_provider=FakeProvider([""]))
    with pytest.raises(RuntimeError, match="empty response"):
        brain.respond("hello")
    assert brain.conversation_history == []


def test_invalid_prompt_and_token_limit_are_rejected():
    brain = TARABrain(llm_provider=FakeProvider())
    with pytest.raises(ValueError, match="prompt"):
        brain.respond("  ")
    with pytest.raises(ValueError, match="between 1 and 4096"):
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


def test_provider_exception_does_not_mutate_conversation_history():
    class RaisingProvider:
        def complete(self, *_args, **_kwargs):
            raise RuntimeError("provider unavailable")

    brain = TARABrain(llm_provider=RaisingProvider())
    brain.conversation_history = [("User", "earlier"), ("TARA", "saved response")]
    before = list(brain.conversation_history)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        brain.respond("new message")
    assert brain.conversation_history == before


@pytest.mark.parametrize("turns", [0, -1, True, 1.5, MAX_HISTORY_TURNS + 1])
def test_history_turn_limit_is_bounded(turns):
    with pytest.raises(ValueError, match="max_history_turns"):
        TARABrain(llm_provider=FakeProvider(), max_history_turns=turns)


def test_oversized_user_message_is_rejected_before_provider_call():
    provider = FakeProvider()
    brain = TARABrain(llm_provider=provider)
    with pytest.raises(ValueError, match="characters"):
        brain.respond("x" * (MAX_USER_MESSAGE_CHARS + 1))
    assert provider.calls == []


def test_assembled_context_is_bounded_by_dropping_oldest_history():
    provider = FakeProvider()
    brain = TARABrain(llm_provider=provider, max_history_turns=3)
    brain.conversation_history = [
        ("User", "old " * 5000),
        ("TARA", "old reply " * 5000),
        ("User", "recent"),
        ("TARA", "recent reply"),
    ]
    brain.respond("latest")
    prompt = provider.calls[0][0]
    assert len(prompt) <= MAX_CONTEXT_CHARS
    assert "User: latest" in prompt
    assert "User: old" not in prompt


def test_oversized_provider_response_does_not_enter_history():
    provider = FakeProvider(["x" * (MAX_PROVIDER_RESPONSE_CHARS + 1)])
    brain = TARABrain(llm_provider=provider)
    with pytest.raises(RuntimeError, match="exceeded"):
        brain.respond("hello")
    assert brain.conversation_history == []


def test_clear_waits_for_inflight_response_and_leaves_context_empty():
    entered = Event()
    release = Event()

    class SlowProvider(FakeProvider):
        def complete(self, prompt, **kwargs):
            entered.set()
            assert release.wait(3)
            return "finished"

    brain = TARABrain(llm_provider=SlowProvider())
    response_thread = Thread(target=lambda: brain.respond("in flight"))
    response_thread.start()
    assert entered.wait(3)

    clear_done = Event()
    clear_thread = Thread(target=lambda: (brain.clear_context(), clear_done.set()))
    clear_thread.start()
    assert not clear_done.wait(0.05)
    release.set()
    response_thread.join(3)
    clear_thread.join(3)
    assert not response_thread.is_alive()
    assert not clear_thread.is_alive()
    assert clear_done.is_set()
    assert brain.conversation_history == []


def test_cli_integer_environment_limits(monkeypatch):
    monkeypatch.setenv("LIMIT", "65")
    with pytest.raises(ValueError, match="between 1 and 64"):
        _positive_int_env("LIMIT", 12, maximum=64)
    monkeypatch.setenv("LIMIT", "4097")
    with pytest.raises(ValueError, match="between 1 and 4096"):
        _positive_int_env("LIMIT", 128, maximum=4096)



def test_hinglish_follow_up_retains_prior_conversation_context():
    provider = FakeProvider(["Hindi mein samjha dunga", "Laptop options bata deta hoon"])
    brain = TARABrain(llm_provider=provider, language="auto", max_history_turns=3)
    brain.respond("Bhai, mujhe Hindi mein samjhana")
    brain.respond("Aur editing ke liye laptop kaunsa?")
    second_prompt = provider.calls[1][0]
    assert "User: Bhai, mujhe Hindi mein samjhana" in second_prompt
    assert "TARA: Hindi mein samjha dunga" in second_prompt
    assert "User: Aur editing ke liye laptop kaunsa?" in second_prompt
