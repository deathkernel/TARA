from types import SimpleNamespace

from src.__main__ import chat_loop


class FakeBrain:
    def __init__(self):
        self.prompts = []

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
    assert "Commands: /help, /exit" in output
