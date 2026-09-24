from types import SimpleNamespace

import pytest

from src.groq_provider import GroqProvider


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="structured answer"))],
            usage=SimpleNamespace(total_tokens=12),
        )


class FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeCompletions())


def test_groq_provider_tracks_usage_and_budget():
    client = FakeClient()
    provider = GroqProvider(client=client, max_daily_requests=1, max_daily_tokens=20)
    assert provider.complete("hello") == "structured answer"
    assert provider.usage.requests == 1
    assert provider.usage.tokens == 12
    assert provider.remaining_daily_requests == 0

    with pytest.raises(RuntimeError, match="daily budget exhausted"):
        provider.complete("second")

    assert client.chat.completions.calls == 1


def test_groq_provider_requires_key_when_no_client(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        GroqProvider()
