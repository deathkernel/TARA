import asyncio
import json

from services.inference import provider as provider_module
from services.inference.provider import GroqStreamingProvider, ProviderUnavailable


class FakeResponse:
    status_code = 200

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def aiter_lines(self):
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
        yield "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]})
        yield "data: " + json.dumps({"choices": [], "usage": {"total_tokens": 9}})
        yield "data: [DONE]"


class FakeClient:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.request = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, *, headers, json):
        self.request = {
            "method": method,
            "url": url,
            "headers": headers,
            "json": json,
        }
        return FakeResponse()


def test_groq_stream_parses_tokens_and_usage(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("TARA_GROQ_MODEL", "openai/gpt-oss-20b")
    monkeypatch.setenv("TARA_GROQ_MAX_COMPLETION_TOKENS", "128")
    monkeypatch.setenv("TARA_GROQ_MAX_DAILY_REQUESTS", "3")
    monkeypatch.setenv("TARA_GROQ_MAX_DAILY_TOKENS", "500")
    fake_client = FakeClient()
    monkeypatch.setattr(provider_module.httpx, "AsyncClient", lambda **kwargs: fake_client)
    provider = GroqStreamingProvider()

    async def collect():
        return [event async for event in provider.stream("hello")]

    events = asyncio.run(collect())

    assert events == [
        {"type": "token", "token": "Hello"},
        {"type": "token", "token": " world"},
    ]
    assert fake_client.request["url"] == provider_module.GROQ_CHAT_COMPLETIONS_URL
    assert fake_client.request["headers"]["Authorization"] == "Bearer test-key"
    assert fake_client.request["json"]["stream"] is True
    assert fake_client.request["json"]["max_completion_tokens"] == 128
    assert provider._usage.tokens == 9
    assert provider._usage.requests == 1


def test_groq_stream_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    provider = GroqStreamingProvider()

    async def collect():
        return [event async for event in provider.stream("hello")]

    try:
        asyncio.run(collect())
    except ProviderUnavailable as exc:
        assert "not configured" in str(exc)
    else:
        raise AssertionError("Expected provider configuration error")


def test_groq_stream_rejects_invalid_budget_configuration(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("TARA_GROQ_MAX_COMPLETION_TOKENS", "0")
    provider = GroqStreamingProvider()

    async def collect():
        return [event async for event in provider.stream("hello")]

    try:
        asyncio.run(collect())
    except ProviderUnavailable as exc:
        assert "outside allowed bounds" in str(exc)
    else:
        raise AssertionError("Expected invalid budget configuration error")
