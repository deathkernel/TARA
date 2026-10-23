from types import SimpleNamespace

from src.__main__ import chat_loop
from src.brain import TARABrain
from src.memory import LongTermMemory


class FakeBrain:
    def __init__(self):
        self.prompts = []
        self.last_emotion = "neutral"
        self.memory = LongTermMemory()

    def respond(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return SimpleNamespace(text=f"reply to {prompt}")


def test_chat_loop_routes_messages_and_exits():
    brain = FakeBrain()
    inputs = iter(["hello TARA", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert brain.prompts == [("hello TARA", {"max_new_tokens": 128, "temperature": 0.6})]
    assert "TARA: reply to hello TARA" in output
    assert output[-1] == "TARA: Bye, bhai!"


def test_help_lists_only_conversation_and_memory_commands():
    brain = FakeBrain()
    inputs = iter(["/help", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    help_line = next(line for line in output if "Commands:" in line)
    assert "/memory [query]" in help_line
    assert "/remember key=value" in help_line
    assert "/forget key" in help_line
    assert "/api " not in help_line
    assert "/terminal" not in help_line


def test_memory_commands_persist_retrieve_and_forget(tmp_path):
    brain = FakeBrain()
    memory_path = tmp_path / "tara_memory.json"
    brain.memory = LongTermMemory(memory_path)
    inputs = iter([
        "/remember preferred_language=Hinglish",
        "/memory preferred_language",
        "/learn editor=DaVinci Resolve",
        "/memory",
        "/forget editor",
        "/exit",
    ])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert any("Saved memory: preferred_language" in line for line in output)
    assert any("Hinglish" in line for line in output)
    assert any("editor: DaVinci Resolve" in line for line in output)
    assert any("Forgot memory: editor" in line for line in output)
    assert brain.prompts == []
    assert memory_path.exists()
    reloaded = LongTermMemory(memory_path)
    assert reloaded.keys() == ["preferred_language"]


def test_memory_commands_reject_empty_and_oversized_values(tmp_path):
    brain = FakeBrain()
    memory_path = tmp_path / "tara_memory.json"
    brain.memory = LongTermMemory(memory_path)
    inputs = iter([
        "/remember =value",
        "/remember key=",
        "/remember " + ("k" * 121) + "=value",
        "/remember key=" + ("v" * 2001),
        "/exit",
    ])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert any("Both memory key and value are required" in line for line in output)
    assert sum("at most" in line for line in output) == 2
    assert brain.memory.keys() == []
    assert not memory_path.exists()
    assert brain.prompts == []


def test_memory_commands_fail_gracefully_when_memory_is_not_configured():
    brain = SimpleNamespace(respond=lambda *args, **kwargs: None, llm_provider=None)
    inputs = iter(["/memory", "/remember language=Hinglish", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert sum("Memory is not configured" in line for line in output) == 2


def test_conversation_context_is_bounded_and_reused():
    calls = []

    class Provider:
        default_max_tokens = 64

        def complete(self, prompt, **kwargs):
            calls.append((prompt, kwargs))
            return f"reply-{len(calls)}"

    brain = TARABrain(llm_provider=Provider(), max_history_turns=2)
    brain.respond("first message")
    brain.respond("second message")
    brain.respond("third message")
    assert len(brain.conversation_history) == 4
    assert "User: second message" in calls[1][0]
    assert "TARA: reply-2" in calls[2][0]
    assert brain.conversation_history[0] == ("User", "second message")
    brain.respond("fourth message")
    assert "User: first message" not in calls[3][0]
    assert "User: second message" in calls[3][0]
    assert "User: third message" in calls[3][0]



def test_invalid_response_token_setting_is_reported_and_chat_continues(monkeypatch):
    monkeypatch.setenv("TARA_MAX_RESPONSE_TOKENS", "0")
    brain = FakeBrain()
    inputs = iter(["hello", "/exit"])
    output = []

    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert brain.prompts == []
    assert any("TARA_MAX_RESPONSE_TOKENS must be a positive integer" in line for line in output)
    assert output[-1] == "TARA: Bye, bhai!"


def test_eof_closes_chat_without_provider_call():
    brain = FakeBrain()
    output = []

    chat_loop(brain, input_fn=lambda _: (_ for _ in ()).throw(EOFError()), output_fn=output.append)

    assert brain.prompts == []
    assert output[-1] == "\nTARA: Chat closed."


def test_display_text_removes_control_characters_and_bounds_output():
    from src.__main__ import _display_text

    assert _display_text("hello\x00\nworld") == "helloworld"
    assert _display_text("x" * 10, limit=5) == "xxxx…"
