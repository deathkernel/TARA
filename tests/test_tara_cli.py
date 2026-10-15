from types import SimpleNamespace

from src.__main__ import chat_loop


class FakeBrain:
    def __init__(self):
        self.prompts = []
        self.last_emotion = "neutral"

    def respond(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return SimpleNamespace(text=f"reply to {prompt}")


class FakeAPIs:
    def __init__(self):
        self.calls = []

    def search_github_repositories(self, query):
        self.calls.append(("github", query))
        return SimpleNamespace(provider="GitHub", source_url="https://api.github.com/search/repositories?q=test",
                               retrieved_at=123.0, trusted=False, data={"items": [{"full_name": "example/repo"}]})

    def search_wikipedia(self, query):
        self.calls.append(("wiki", query))
        return SimpleNamespace(provider="Wikipedia", source_url="https://en.wikipedia.org/w/api.php",
                               retrieved_at=123.0, trusted=False, data=["test", ["Test"], [""], ["https://en.wikipedia.org/wiki/Test"]])

    def get_weather(self, city):
        self.calls.append(("weather", city))
        return SimpleNamespace(provider="Open-Meteo", source_url="https://api.open-meteo.com/v1/forecast",
                               retrieved_at=123.0, trusted=False, data={"location": {"name": city}, "weather": {"current": {}}})


def test_chat_loop_routes_messages_and_exits():
    brain = FakeBrain()
    inputs = iter(["hello TARA", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=FakeAPIs())
    assert brain.prompts == [("hello TARA", {"max_new_tokens": 128, "temperature": 0.6})]
    assert "TARA: reply to hello TARA" in output
    assert output[-1] == "TARA: Bye, bhai!"


def test_help_lists_non_persistent_and_public_api_commands():
    brain = FakeBrain()
    inputs = iter(["", "/help", "/emotion", "/quit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=FakeAPIs())
    assert brain.prompts == []
    help_line = next(line for line in output if "Commands:" in line)
    assert "/learn" not in help_line and "/memory" not in help_line
    assert "/api github" in help_line and "/api weather" in help_line
    assert any("Last detected emotional cue" in line for line in output)


def test_memory_commands_are_disabled():
    brain = FakeBrain()
    inputs = iter(["/learn preferred_language=Hinglish", "/memory", "/forget project", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=FakeAPIs())
    assert all("Persistent conversation memory" in line for line in output if "removed" in line)
    assert brain.prompts == []


def test_api_commands_use_injected_public_apis_without_calling_llm():
    brain, apis = FakeBrain(), FakeAPIs()
    inputs = iter(["/api github open source agents", "/api wiki artificial intelligence", "/api weather Pune", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=apis)
    assert apis.calls == [
        ("github", "open source agents"),
        ("wiki", "artificial intelligence"),
        ("weather", "Pune"),
    ]
    assert brain.prompts == []
    assert any('"trusted": false' in line for line in output)
    assert any("source_url" in line for line in output)


def test_api_commands_validate_provider_and_missing_query():
    brain, apis = FakeBrain(), FakeAPIs()
    inputs = iter(["/api", "/api unsupported query", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=apis)
    assert any("Usage:" in line for line in output)
    assert any("Unknown API provider" in line for line in output)
    assert apis.calls == []
