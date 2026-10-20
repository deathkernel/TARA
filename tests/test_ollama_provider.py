import json
import urllib.error

import pytest

from src.ollama_provider import OllamaProvider, OllamaProviderError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_complete_sends_chat_payload_and_returns_text(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"message": {"content": "Hello from Ollama"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider(model="test-model", host="http://localhost:11434")

    result = provider.complete(
        "Hi",
        system="Be concise.",
        max_completion_tokens=64,
        temperature=0.4,
    )

    assert result == "Hello from Ollama"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["timeout"] == 180
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]
    assert captured["payload"]["options"] == {
        "num_predict": 64,
        "temperature": 0.4,
    }


def test_complete_reports_connection_errors(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = OllamaProvider()

    with pytest.raises(OllamaProviderError, match="Cannot connect to Ollama"):
        provider.complete("Hi")


def test_complete_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt must not be empty"):
        OllamaProvider().complete("  ")


def test_complete_rejects_invalid_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"unexpected": "shape"}),
    )
    with pytest.raises(OllamaProviderError, match="message.content"):
        OllamaProvider().complete("Hi")
