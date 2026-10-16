from dataclasses import dataclass
from types import SimpleNamespace

from src.public_api_tools import APIObservation
from src.ollama_tool_calling import OllamaToolCalling


@dataclass
class FakeMessage:
    content: str = ""
    tool_calls: list | None = None

    def model_dump(self, exclude_none=True):
        return {
            "role": "assistant",
            "content": self.content,
            **({"tool_calls": self.tool_calls} if self.tool_calls is not None else {}),
        }


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(message=self.responses.pop(0))


class FakePublicTools:
    def __init__(self):
        self.calls = []

    def get_weather(self, city):
        self.calls.append(("weather", city))
        return APIObservation(
            provider="Open-Meteo",
            data={"location": city, "temperature_c": 29},
            source_url="https://api.open-meteo.com/v1/forecast",
            retrieved_at=1234.0,
            trusted=False,
        )

    def search_github_repositories(self, query, *, limit=5):
        self.calls.append(("github", query, limit))
        return APIObservation("GitHub", {"items": []}, "https://api.github.com/search/repositories", 1234.0)

    def search_wikipedia(self, query, *, limit=5):
        self.calls.append(("wiki", query, limit))
        return APIObservation("Wikipedia", [], "https://en.wikipedia.org/w/api.php", 1234.0)

    def search_hacker_news(self, query, *, limit=5):
        self.calls.append(("hn", query, limit))
        return APIObservation("Hacker News", {"hits": []}, "https://hn.algolia.com/api/v1/search", 1234.0)

    def search_stackoverflow(self, query, *, limit=5):
        self.calls.append(("stackoverflow", query, limit))
        return APIObservation("Stack Exchange", {"items": []}, "https://api.stackexchange.com/2.3/search/advanced", 1234.0)


def tool_call(name, arguments):
    return {"function": {"name": name, "arguments": arguments}}


def test_weather_tool_call_is_validated_executed_and_returned_to_model():
    public_tools = FakePublicTools()
    client = FakeClient([
        FakeMessage(tool_calls=[tool_call("get_weather", {"city": "Pune"})]),
        FakeMessage(content="Pune is currently 29°C."),
    ])
    runner = OllamaToolCalling(
        model="llama3.2:3b",
        client=client,
        public_tools=public_tools,
    )

    result = runner.complete("What's the weather in Pune?")

    assert result == "Pune is currently 29°C."
    assert public_tools.calls == [("weather", "Pune")]
    assert client.calls[0]["model"] == "llama3.2:3b"
    assert client.calls[0]["stream"] is False
    messages = client.calls[1]["messages"]
    tool_message = next(message for message in messages if message.get("role") == "tool")
    assert "UNTRUSTED EXTERNAL DATA" in tool_message["content"]
    assert "https://api.open-meteo.com/v1/forecast" in tool_message["content"]


def test_unknown_model_requested_tool_is_never_executed():
    public_tools = FakePublicTools()
    client = FakeClient([
        FakeMessage(tool_calls=[tool_call("run_shell", {"command": "whoami"})]),
        FakeMessage(content="I cannot run shell commands."),
    ])
    runner = OllamaToolCalling(model="llama3.2:3b", client=client, public_tools=public_tools)

    assert runner.complete("Run a shell command") == "I cannot run shell commands."
    assert public_tools.calls == []
    tool_message = next(message for message in client.calls[1]["messages"] if message.get("role") == "tool")
    assert "not allowlisted" in tool_message["content"]


def test_invalid_arguments_are_rejected_before_public_tool_execution():
    public_tools = FakePublicTools()
    runner = OllamaToolCalling(model="llama3.2:3b", client=FakeClient([]), public_tools=public_tools)

    assert "unexpected argument fields" in runner._dispatch(
        "get_weather", {"city": "Pune", "url": "https://example.com"}
    )
    assert "limit must be an integer" in runner._dispatch(
        "search_github", {"query": "TARA", "limit": True}
    )
    assert "non-empty string" in runner._dispatch("get_weather", {"city": 123})
    assert public_tools.calls == []


def test_tool_results_are_bounded():
    class LargePublicTools(FakePublicTools):
        def get_weather(self, city):
            return APIObservation(
                "Open-Meteo",
                {"text": "x" * 20_000},
                "https://api.open-meteo.com/v1/forecast",
                1234.0,
            )

    runner = OllamaToolCalling(
        model="llama3.2:3b",
        client=FakeClient([]),
        public_tools=LargePublicTools(),
    )
    result = runner._dispatch("get_weather", {"city": "Pune"})
    assert len(result) <= runner.MAX_TOOL_RESULT_CHARS
    assert "truncated" in result


def test_tool_call_loop_has_a_hard_round_limit():
    responses = [
        FakeMessage(tool_calls=[tool_call("search_wikipedia", {"query": "TARA"})]),
        FakeMessage(tool_calls=[tool_call("search_wikipedia", {"query": "TARA"})]),
        FakeMessage(content="Done after the bounded tool loop."),
    ]
    public_tools = FakePublicTools()
    client = FakeClient(responses)
    runner = OllamaToolCalling(
        model="llama3.2:3b",
        client=client,
        public_tools=public_tools,
        max_tool_rounds=2,
    )

    assert runner.complete("Research TARA") == "Done after the bounded tool loop."
    assert len(client.calls) == 3
    assert public_tools.calls == [("wiki", "TARA", 5), ("wiki", "TARA", 5)]


def test_tool_calling_rejects_invalid_round_configuration():
    try:
        OllamaToolCalling(model="llama3.2:3b", client=FakeClient([]), max_tool_rounds=4)
    except ValueError as exc:
        assert "max_tool_rounds" in str(exc)
    else:
        raise AssertionError("Expected max_tool_rounds to be bounded")
