"""Focused validation tests for the local web API."""
from fastapi.testclient import TestClient
from src.web_api import app

client = TestClient(app)

def test_health_reports_provider_configuration():
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "Ollama (local)"
    assert payload["model"]

def test_chat_rejects_whitespace_message():
    response = client.post("/api/chat", json={"message": "   ", "history": []})
    assert response.status_code == 422

def test_chat_rejects_empty_history_content():
    response = client.post("/api/chat", json={"message": "hello", "history": [{"role": "user", "content": ""}]})
    assert response.status_code == 422
