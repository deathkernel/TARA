from src.llm_provider_factory import provider_from_env
from src.ollama_provider import OllamaProvider


def test_factory_defaults_to_existing_tara_language_model(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARA_LLM_BACKEND", raising=False)
    assert provider_from_env() is None


def test_factory_selects_local_ollama_provider(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARA_LLM_BACKEND", "ollama")
    monkeypatch.setenv("TARA_LLAMA_MODEL", "llama3.2:1b")
    provider = provider_from_env()
    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2:1b"


def test_factory_loads_ollama_configuration_from_dotenv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TARA_LLM_BACKEND", raising=False)
    monkeypatch.delenv("TARA_LLAMA_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        "TARA_LLM_BACKEND=ollama\nTARA_LLAMA_MODEL=llama3.2:1b\n",
        encoding="utf-8",
    )

    provider = provider_from_env()

    assert isinstance(provider, OllamaProvider)
    assert provider.model == "llama3.2:1b"


def test_factory_rejects_unknown_backend(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TARA_LLM_BACKEND", "unknown")
    try:
        provider_from_env()
    except ValueError as exc:
        assert "Unsupported TARA_LLM_BACKEND" in str(exc)
    else:
        raise AssertionError("Expected unknown backend to be rejected")
