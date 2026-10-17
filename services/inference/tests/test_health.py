from fastapi.testclient import TestClient

from services.inference.app import app

client = TestClient(app)


def test_liveness_reports_process_health() -> None:
    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "tara-inference"}


def test_compatibility_health_route_is_live() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_fails_closed_until_inference_is_connected() -> None:
    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"
    assert response.json()["reason"] == "inference adapter is not connected yet"
