# Ollama provider configuration

TARA's Ollama adapter calls the local Ollama chat endpoint and keeps provider
failures explicit. It does not train or fine-tune the model.

## Quick start

1. Install Ollama and pull a model, for example `ollama pull llama3.2:3b`.
2. Start Ollama, then run TARA with `python -m src`.
3. Use `/help` in the chat for supported runtime commands.

## Environment settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `TARA_LLM_BACKEND` | `ollama` in the chat entry point | Select the configured provider |
| `TARA_LLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama server base URL |
| `TARA_LLM_TIMEOUT_SECONDS` | `120` | Request timeout, in seconds |
| `TARA_POWER_MODE` | `eco` | Resource profile: `eco`, `balanced`, or `performance` |
| `TARA_OLLAMA_NUM_CTX` | Profile-specific | Context window size |
| `TARA_OLLAMA_NUM_THREADS` | Profile/CPU-specific | CPU threads; zero means automatic |
| `TARA_OLLAMA_NUM_BATCH` | Profile-specific | Prompt processing batch size |
| `TARA_OLLAMA_KEEP_ALIVE` | Profile-specific | How long Ollama keeps the model loaded |
| `TARA_MAX_RESPONSE_TOKENS` | Provider profile | Chat response token cap |

## Resource profiles

- **eco** — conservative defaults for everyday desktop use.
- **balanced** — larger context and response budget.
- **performance** — highest configured context and response budget; may use more RAM/VRAM.

Choose based on available hardware. Higher values do not guarantee better answers
and may increase memory use or latency.

## Failure behavior

The adapter validates configuration before sending requests and raises readable
errors for unreachable servers, HTTP failures, Ollama error payloads, malformed
responses, and invalid generation parameters. Automated tests use an injected
transport, so ordinary provider tests do not need a running Ollama server.

Never commit real API keys or private environment files to the repository.
