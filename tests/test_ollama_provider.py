import pytest

from src.ollama_provider import MAX_BATCH_SIZE, MAX_COMPLETION_TOKENS, MAX_CONTEXT_TOKENS, MAX_CPU_THREADS, OllamaProvider, _http_json


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


@pytest.mark.parametrize("timeout", [True, float("nan"), float("inf"), 0, -1, 3601, 10**1000, "30"])
def test_ollama_provider_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="timeout must be finite"):
        OllamaProvider(timeout=timeout)


@pytest.mark.parametrize("kwargs", [
    {"model": None},
    {"model": 123},
    {"base_url": None},
    {"base_url": 123},
    {"power_mode": None},
    {"num_ctx": True},
    {"num_ctx": 1.5},
    {"num_thread": True},
    {"num_thread": -1},
    {"num_batch": False},
    {"num_batch": 0},
    {"num_ctx": MAX_CONTEXT_TOKENS + 1},
    {"num_thread": MAX_CPU_THREADS + 1},
    {"num_batch": MAX_BATCH_SIZE + 1},
    {"keep_alive": 123},
    {"keep_alive": ""},
])
def test_ollama_provider_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        OllamaProvider(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"max_completion_tokens": True},
    {"max_completion_tokens": 1.5},
    {"max_completion_tokens": "20"},
    {"max_completion_tokens": 0},
    {"temperature": True},
    {"temperature": float("nan")},
    {"temperature": float("inf")},
    {"temperature": -0.1},
    {"temperature": 2.1},
    {"system": 123},
])
def test_ollama_provider_rejects_invalid_completion_options(kwargs):
    provider = OllamaProvider(transport=lambda *_: {"message": {"content": "ok"}})
    with pytest.raises(ValueError):
        provider.complete("Hello", **kwargs)


def test_ollama_provider_rejects_non_object_transport_response():
    provider = OllamaProvider(transport=lambda *_: ["unexpected"])
    with pytest.raises(RuntimeError, match="invalid JSON response"):
        provider.complete("Hello")


@pytest.mark.parametrize("base_url", [
    "ftp://127.0.0.1:11434",
    "file:///etc/passwd",
    "http:///missing-host",
    "http://user:password@127.0.0.1:11434",
    "http://127.0.0.1:11434/path",
    "http://127.0.0.1:11434/?query=1",
    "http://127.0.0.1:11434/#fragment",
    "http://127.0.0.1:99999",
])
def test_ollama_provider_rejects_unsafe_or_malformed_base_url(base_url):
    with pytest.raises(ValueError, match="base_url"):
        OllamaProvider(base_url=base_url)


def test_ollama_provider_caps_completion_tokens():
    provider = OllamaProvider(transport=lambda *_: {"message": {"content": "ok"}})
    with pytest.raises(ValueError, match="max_completion_tokens"):
        provider.complete("Hello", max_completion_tokens=MAX_COMPLETION_TOKENS + 1)


def test_http_json_rejects_oversized_response(monkeypatch):
    import src.ollama_provider as module

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            assert size == module.MAX_RESPONSE_BYTES + 1
            return b"x" * size

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setattr(module, "build_opener", lambda *_: FakeOpener())
    with pytest.raises(RuntimeError, match="8 MiB safety limit"):
        _http_json("http://127.0.0.1:11434/api/chat", {}, 1.0)


def test_http_json_rejects_malformed_json(monkeypatch):
    import src.ollama_provider as module

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self, size=-1):
            return b"not-json"

    class FakeOpener:
        def open(self, request, timeout):
            return FakeResponse()

    monkeypatch.setattr(module, "build_opener", lambda *_: FakeOpener())
    with pytest.raises(RuntimeError, match="malformed JSON"):
        _http_json("http://127.0.0.1:11434/api/chat", {}, 1.0)


@pytest.mark.parametrize("response", [
    {},
    {"message": None},
    {"message": []},
    {"message": {"content": None}},
    {"message": {"content": 123}},
])
def test_ollama_provider_rejects_missing_or_non_text_message_content(response):
    provider = OllamaProvider(transport=lambda *_: response)
    with pytest.raises(RuntimeError, match="message.content"):
        provider.complete("Hello")


def test_no_redirect_handler_refuses_redirect_targets():
    from src.ollama_provider import _NoRedirectHandler

    handler = _NoRedirectHandler()
    redirect = handler.redirect_request(
        None,
        None,
        302,
        "Found",
        {},
        "https://example.com/collect",
    )

    assert redirect is None


def test_http_json_configures_no_redirect_handler(monkeypatch):
    import src.ollama_provider as module

    configured = []

    class FakeOpener:
        def open(self, request, timeout):
            raise AssertionError("unexpected request execution")

    def fake_build_opener(*handlers):
        configured.extend(handlers)
        return FakeOpener()

    monkeypatch.setattr(module, "build_opener", fake_build_opener)
    with pytest.raises(AssertionError, match="unexpected request execution"):
        _http_json("http://127.0.0.1:11434/api/chat", {}, 1.0)

    assert len(configured) == 1
    assert isinstance(configured[0], module._NoRedirectHandler)
