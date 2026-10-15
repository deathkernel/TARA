from types import SimpleNamespace

from src.__main__ import chat_loop


class FakeBrain:
    def __init__(self):
        self.prompts = []
        self.last_emotion = "neutral"

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


def test_help_lists_only_non_persistent_commands():
    brain = FakeBrain()
    inputs = iter(["", "/help", "/emotion", "/quit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert brain.prompts == []
    help_line = next(line for line in output if "Commands:" in line)
    assert "/learn" not in help_line and "/memory" not in help_line
    assert any("Last detected emotional cue" in line for line in output)


def test_memory_commands_are_disabled():
    brain = FakeBrain()
    inputs = iter(["/learn preferred_language=Hinglish", "/memory", "/forget project", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)
    assert all("Persistent conversation memory" in line for line in output if "removed" in line)
    assert brain.prompts == []
