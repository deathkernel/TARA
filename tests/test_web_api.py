from types import SimpleNamespace

from fastapi.testclient import TestClient

from src import web_api


class FakeProvider:
    def complete(self, prompt, **kwargs):
        return "TARA test response: " + prompt


def test_health_does_not_require_provider_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = TestClient(web_api.app).get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["provider_configured"] is False


def test_chat_rejects_missing_provider_configuration(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    response = TestClient(web_api.app).post("/api/chat", json={"message": "hello"})
    assert response.status_code == 503
    assert "not configured" in response.json()["detail"]


def test_chat_uses_provider_and_returns_message(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-only-key")
    web_api._provider.cache_clear()
    monkeypatch.setattr(web_api, "_provider", lambda: FakeProvider())
    response = TestClient(web_api.app).post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json() == {"message": "TARA test response: hello"}


def test_chat_rejects_blank_message():
    response = TestClient(web_api.app).post("/api/chat", json={"message": "  "})
    assert response.status_code == 422
