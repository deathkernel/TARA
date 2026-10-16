from types import SimpleNamespace

from src.__main__ import _run_api_command, _summarize_api_observation, chat_loop
from src.memory import LongTermMemory


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
                               retrieved_at=123.0, trusted=False, data={"items": [{"full_name": "example/repo", "description": "A test repository", "stargazers_count": 7, "html_url": "https://github.com/example/repo"}]})

    def search_wikipedia(self, query):
        self.calls.append(("wiki", query))
        return SimpleNamespace(provider="Wikipedia", source_url="https://en.wikipedia.org/w/api.php",
                               retrieved_at=123.0, trusted=False, data=["test", ["Test"], [""], ["https://en.wikipedia.org/wiki/Test"]])

    def search_hacker_news(self, query):
        self.calls.append(("hn", query))
        return SimpleNamespace(provider="Hacker News", source_url="https://hn.algolia.com/api/v1/search",
                               retrieved_at=123.0, trusted=False, data={"hits": [{"title": query}]})

    def search_stackoverflow(self, query):
        self.calls.append(("stackoverflow", query))
        return SimpleNamespace(provider="Stack Exchange", source_url="https://api.stackexchange.com/2.3/search/advanced",
                               retrieved_at=123.0, trusted=False, data={"items": [{"title": query}]})

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
    assert "/memory [query]" in help_line and "/remember key=value" in help_line
    assert "/forget key" in help_line
    assert "/api github" in help_line and "/api weather" in help_line
    assert "/api hn" in help_line and "/api stackoverflow" in help_line
    assert any("Last detected emotional cue" in line for line in output)


def test_memory_commands_persist_retrieve_and_forget(tmp_path):
    brain = FakeBrain()
    memory_path = tmp_path / "tara_memory.json"
    brain.memory = LongTermMemory(memory_path)
    inputs = iter([
        "/remember preferred_language=Hinglish",
        "/memory language",
        "/learn editor=DaVinci Resolve",
        "/memory",
        "/forget editor",
        "/exit",
    ])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=FakeAPIs())

    assert any("Saved memory: preferred_language" in line for line in output)
    assert any("Hinglish" in line for line in output)
    assert any("editor: DaVinci Resolve" in line for line in output)
    assert any("Forgot memory: editor" in line for line in output)
    assert brain.prompts == []
    assert memory_path.exists()
    reloaded = LongTermMemory(memory_path)
    assert reloaded.keys() == ["preferred_language"]


def test_memory_commands_fail_gracefully_when_memory_is_not_configured():
    brain = FakeBrain()
    inputs = iter(["/memory", "/remember language=Hinglish", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=FakeAPIs())
    assert sum("Memory is not configured" in line for line in output) == 2
    assert brain.prompts == []


def test_api_commands_use_injected_public_apis_without_calling_llm():
    brain, apis = FakeBrain(), FakeAPIs()
    inputs = iter(["/api github open source agents", "/api wiki artificial intelligence", "/api weather Pune", "/api hn AI agents", "/api stackoverflow Python async", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=apis)
    assert apis.calls == [
        ("github", "open source agents"),
        ("wiki", "artificial intelligence"),
        ("weather", "Pune"),
        ("hn", "AI agents"),
        ("stackoverflow", "Python async"),
    ]
    assert brain.prompts == []
    assert any('"trusted": false' in line for line in output)
    assert any("source_url" in line for line in output)
    assert any("Found 1 repositories." in line for line in output)
    assert any("example/repo" in line for line in output)
    assert any("untrusted external data" in line for line in output)
    assert any("was not sent to the language model" in line for line in output)


def test_api_commands_validate_provider_and_missing_query():
    brain, apis = FakeBrain(), FakeAPIs()
    inputs = iter(["/api", "/api unsupported query", "/exit"])
    output = []
    chat_loop(brain, input_fn=lambda _: next(inputs), output_fn=output.append, api_tools=apis)
    assert any("Usage:" in line for line in output)
    assert any("Unknown API provider" in line for line in output)
    assert apis.calls == []



def test_api_summary_formats_weather_and_removes_terminal_controls():
    observation = SimpleNamespace(
        provider="Open-Meteo",
        source_url="https://api.open-meteo.com/v1/forecast",
        retrieved_at=123.0,
        trusted=False,
        data={
            "location": {"name": "Pune\x1b[31m", "admin1": "Maharashtra", "country": "India"},
            "weather": {"current": {"temperature_2m": 31.5, "relative_humidity_2m": 45, "wind_speed_10m": 8}},
        },
    )
    summary = _summarize_api_observation(observation)
    assert "temperature: 31.5°C" in summary
    assert "humidity: 45%" in summary
    assert "wind: 8 km/h" in summary
    assert "untrusted external data" in summary
    assert "\x1b" not in summary


def test_api_summary_formats_hacker_news_titles_and_links():
    observation = SimpleNamespace(
        provider="Hacker News",
        source_url="https://hn.algolia.com/api/v1/search",
        retrieved_at=123.0,
        trusted=False,
        data={"hits": [{"title": "A story", "url": "https://example.com/story", "points": 12}]},
    )
    summary = _summarize_api_observation(observation)
    assert "Found 1 Hacker News matches." in summary
    assert "A story (points: 12)" in summary
    assert "https://example.com/story" in summary



def test_api_summary_formats_wikipedia_and_stackexchange_results():
    wiki = SimpleNamespace(
        provider="Wikipedia",
        source_url="https://en.wikipedia.org/w/api.php",
        retrieved_at=123.0,
        trusted=False,
        data=["query", ["TARA", "AI"], ["", ""], ["https://en.wikipedia.org/wiki/TARA", "https://en.wikipedia.org/wiki/AI"]],
    )
    stack = SimpleNamespace(
        provider="Stack Exchange",
        source_url="https://api.stackexchange.com/2.3/search/advanced",
        retrieved_at=123.0,
        trusted=False,
        data={"items": [{"title": "How to test Python?", "link": "https://stackoverflow.com/q/1", "score": 4}]},
    )
    wiki_summary = _summarize_api_observation(wiki)
    stack_summary = _summarize_api_observation(stack)
    assert "Found 2 Wikipedia matches." in wiki_summary
    assert "https://en.wikipedia.org/wiki/TARA" in wiki_summary
    assert "Found 1 Stack Overflow questions." in stack_summary
    assert "How to test Python? (score: 4)" in stack_summary
    assert "https://stackoverflow.com/q/1" in stack_summary


def test_api_summary_bounds_untrusted_fields():
    observation = SimpleNamespace(
        provider="Hacker News",
        source_url="https://hn.algolia.com/api/v1/search",
        retrieved_at=123.0,
        trusted=False,
        data={"hits": [{"title": "T" * 500, "url": "https://example.com"}]},
    )
    summary = _summarize_api_observation(observation)
    assert "T" * 200 not in summary
    assert "T" * 159 + "…" in summary



def test_large_api_output_stays_valid_json_when_truncated():
    class LargeAPI:
        def search_github_repositories(self, query):
            return SimpleNamespace(
                provider="GitHub",
                source_url="https://api.github.com/search/repositories",
                retrieved_at=123.0,
                trusted=False,
                data={"items": [{"full_name": "repo", "description": "x" * 12000}]},
            )

    result = _run_api_command("/api github test", LargeAPI())
    assert result is not None
    assert "Raw response (trusted=false):" in result
    raw = result.split("Raw response (trusted=false):", 1)[1].strip()
    import json
    payload = json.loads(raw)
    assert payload["truncated"] is True
    assert payload["data"]["truncated"] is True
    assert len(raw) < 5000
