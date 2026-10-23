from src.ollama_provider import OllamaProvider
from src.provider_contract import LanguageProvider


def test_ollama_provider_satisfies_language_provider_contract():
    provider = OllamaProvider(transport=lambda *_: {"message": {"content": "hello"}})
    assert isinstance(provider, LanguageProvider)
    assert provider.default_max_tokens == 128
    assert provider.complete("prompt") == "hello"
