from types import SimpleNamespace

from src.__main__ import chat_loop
from src.memory import LongTermMemory


class FakeBrain:
    def __init__(self, memory=None):
        self.prompts = []
        self.memory = memory or LongTermMemory()

    def respond(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return SimpleNamespace(text=f"reply to {prompt}")


def test_chat_loop_routes_user_messages_through_tara():
    brain = FakeBrain()
    inputs = iter(["hello TARA", "/exit"])
    output = []

    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert brain.prompts == [
        ("hello TARA", {"max_new_tokens": 256, "temperature": 0.6})
    ]
    assert "TARA: reply to hello TARA" in output
    assert output[-1] == "TARA: Bye, bhai!"


def test_chat_loop_skips_empty_input_and_handles_help():
    brain = FakeBrain()
    inputs = iter(["", "/help", "/quit"])
    output = []

    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert brain.prompts == []
    assert any("Commands:" in line and "/learn" in line for line in output)


def test_learning_command_persists_and_memory_survives_reload(tmp_path):
    path = tmp_path / "memory.json"
    brain = FakeBrain(LongTermMemory(path=path))
    inputs = iter(["/learn preferred_language=Hinglish", "/exit"])
    output = []

    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert any("Learned and saved 'preferred_language'" in line for line in output)
    reloaded = LongTermMemory(path=path)
    assert reloaded.recall("preferred_language") == "Hinglish"


def test_memory_list_and_forget_commands(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path=path)
    memory.remember("project", "TARA", importance=1.0)
    memory.save()
    brain = FakeBrain(LongTermMemory(path=path))
    inputs = iter(["/memory", "/forget project", "/memory", "/exit"])
    output = []

    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append)

    assert any("project: TARA" in line for line in output)
    assert any("Forgotten 'project'." in line for line in output)
    assert "TARA: My long-term memory is empty." in output
    assert LongTermMemory(path=path).keys() == []
