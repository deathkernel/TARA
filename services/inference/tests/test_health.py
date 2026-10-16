from fastapi.testclient import TestClient

from services.inference.app import app
from services.inference.provider import provider

client = TestClient(app)


def test_liveness_reports_process_health() -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tara-inference"}


def test_compatibility_health_route_is_live() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_fails_closed_without_provider_or_internal_token(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("TARA_INFERENCE_API_TOKEN", raising=False)

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_reports_configured_provider(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setenv("TARA_INFERENCE_API_TOKEN", "test-internal-token")

    response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_stream_requires_internal_bearer_token(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setenv("TARA_INFERENCE_API_TOKEN", "test-internal-token")

    response = client.post("/v1/chat/stream", json={"message": "hello"})

    assert response.status_code == 401


def test_stream_returns_provider_tokens_and_done_event(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setenv("TARA_INFERENCE_API_TOKEN", "test-internal-token")

    async def fake_stream(message: str):
        assert message == "hello TARA"
        yield {"type": "token", "token": "Hello "}
        yield {"type": "token", "token": "there"}

    monkeypatch.setattr(provider, "stream", fake_stream)
    response = client.post(
        "/v1/chat/stream",
        headers={"Authorization": "Bearer test-internal-token"},
        json={"message": "hello TARA"},
    )

    assert response.status_code == 200
    assert '"type": "token", "token": "Hello "' in response.text
    assert '"type": "token", "token": "there"' in response.text
    assert '"type": "done"' in response.text


def test_stream_rejects_extra_fields(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-provider-key")
    monkeypatch.setenv("TARA_INFERENCE_API_TOKEN", "test-internal-token")

    response = client.post(
        "/v1/chat/stream",
        headers={"Authorization": "Bearer test-internal-token"},
        json={"message": "hello", "api_key": "do-not-accept"},
    )

    assert response.status_code == 422
