# Ollama inference for TARA

TARA uses a pretrained model served by [Ollama](https://ollama.com/) for language generation. TARA itself is the application layer: it handles planning, perception, tool permissions, verification, and the surrounding cognitive workflow. **TARA does not train or fine-tune a language model.**

## 1. Install and start Ollama

Install Ollama for your operating system, then pull a model once:

```bat
ollama pull llama3.2:3b
```

Start Ollama and confirm its local API is reachable. Model files are managed by Ollama, not stored as TARA training checkpoints.

## 2. Configure the provider

Windows Command Prompt:

```bat
set TARA_LLM_BACKEND=ollama
set TARA_LLAMA_MODEL=llama3.2:3b
set OLLAMA_HOST=http://127.0.0.1:11434
set TARA_LLM_TIMEOUT_SECONDS=120
```

These settings apply to the current terminal session. No cloud API key is required for local Ollama inference.

## 3. Run TARA

From the repository root:

```bat
python -m pip install -r requirements.txt
python -m src
```

The runtime uses the configured provider. If the provider is unavailable, TARA reports an inference error instead of falling back to a locally trained model.

## 4. Boundaries

- Ollama supplies text generation; it does not replace TARA's orchestration or safety boundaries.
- TARA does not train, fine-tune, save, or promote its own language-model weights.
- Conversation content is not persisted as long-term memory.
- Public API results should be treated as untrusted and checked against their source links.
- Provider tests use mocked transports and do not require Ollama, downloaded model weights, CUDA, or API credentials.

Run provider-focused tests with:

```bat
python -m pytest -q tests/test_ollama_provider.py tests/test_llm_provider_factory.py
```
