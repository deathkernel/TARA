from src.ollama_provider import OllamaProvider


def test_ollama_provider_sends_chat_request_and_returns_text():
    calls = []

    def transport(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {"message": {"role": "assistant", "content": "Hello from Llama"}}

    provider = OllamaProvider(
        model="llama3.2:3b",
        base_url="http://localhost:11434/",
        timeout=5,
        transport=transport,
    )

    result = provider.complete(
        "Hello TARA",
        system="Be useful",
        max_completion_tokens=48,
        temperature=0.2,
    )

    assert result == "Hello from Llama"
    assert calls == [(
        "http://localhost:11434/api/chat",
        {
            "model": "llama3.2:3b",
            "messages": [
                {"role": "system", "content": "Be useful"},
                {"role": "user", "content": "Hello TARA"},
            ],
            "stream": False,
            "keep_alive": "2m",
            "options": {
                "temperature": 0.2,
                "num_predict": 48,
                "num_ctx": 2048,
                "num_thread": 4,
                "num_batch": 64,
            },
        },
        5.0,
    )]


def test_ollama_provider_eco_profile_bounds_resources():
    provider = OllamaProvider(transport=lambda *_: {"message": {"content": "ok"}})
    assert provider.power_mode == "eco"
    assert provider.num_ctx == 2048
    assert provider.num_thread == 4
    assert provider.num_batch == 64
    assert provider.default_max_tokens == 128
    assert provider.keep_alive == "2m"


def test_ollama_provider_rejects_unknown_power_profile():
    try:
        OllamaProvider(power_mode="unlimited")
    except ValueError as exc:
        assert "TARA_POWER_MODE" in str(exc)
    else:
        raise AssertionError("Expected an unsupported power profile to be rejected")


def test_ollama_provider_rejects_empty_prompt():
    provider = OllamaProvider(transport=lambda *_: {})
    try:
        provider.complete("   ")
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("Expected an empty prompt to be rejected")


def test_ollama_provider_reports_server_error():
    provider = OllamaProvider(
        transport=lambda *_: {"error": "model not found"}
    )
    try:
        provider.complete("Hello")
    except RuntimeError as exc:
        assert "model not found" in str(exc)
    else:
        raise AssertionError("Expected an Ollama error to be reported")


def test_ollama_provider_loads_environment(monkeypatch):
    monkeypatch.setenv("TARA_LLAMA_MODEL", "llama3.2:1b")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("TARA_LLM_TIMEOUT_SECONDS", "30")
    provider = OllamaProvider.from_env()
    assert provider.model == "llama3.2:1b"
    assert provider.base_url == "http://127.0.0.1:11434"
    assert provider.timeout == 30.0
