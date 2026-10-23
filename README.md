# TARA — Conversation AI

TARA is a conversation-first AI application. Its current scope is text chat with a local language model, bounded multi-turn context, and explicit user-controlled memory.

## Scope

**Included**
- Interactive chat through `python -m src`
- Local inference through [Ollama](https://ollama.com/)
- Multi-turn conversation context held in the running session
- Optional, explicit persistent memory commands
- Resource-bounded provider configuration and regression tests

**Intentionally out of scope**
- Training or fine-tuning models, dataset collection, tokenizer research, and training experiments
- Autonomous agents, task scheduling, planning/action loops, terminal/file/browser automation, and tool execution
- Autonomous web research and public API command integrations
- Vision, screen perception, audio perception, and voice synthesis
- Self-improvement labs, architecture search, and scientific experiment runners

Those capabilities may be developed as separate future projects. They are not part of the TARA conversation runtime.

## Quick start

1. Install Python 3.11 or newer.
2. Install [Ollama](https://ollama.com/) and start its local service.
3. Pull a model:
   ```bash
   ollama pull llama3.2:3b
   ```
4. Install TARA's test dependency and start chat:
   ```bash
   python -m pip install -r requirements.txt
   python -m src
   ```

TARA defaults to the local Ollama backend. It does not download models or start Ollama for you. See [`.env.example`](.env.example) for optional settings.

## Chat commands

- `/help` — show available commands
- `/exit` — end the chat
- `/remember key=value` — explicitly save a memory
- `/memory [query]` — list memories or search them
- `/forget key` — delete a saved memory

Memory is stored locally at `data/tara_memory.json` by default. Set `TARA_MEMORY_PATH` to change the location. Saved memories are separate from the active conversation context. Do not store passwords, API keys, or other secrets in memory.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `TARA_LLM_BACKEND` | `ollama` | Local provider selection |
| `TARA_LLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `TARA_LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `TARA_POWER_MODE` | `eco` | Resource profile: `eco`, `balanced`, or `performance` |
| `TARA_MEMORY_PATH` | `data/tara_memory.json` | Persistent memory file |
| `TARA_MAX_RESPONSE_TOKENS` | Provider default | Maximum response tokens for chat |

The provider validates its endpoint and resource limits. The default endpoint is local-only; do not expose Ollama to an untrusted network.

## Development and validation

Run the conversation-focused tests and static repository audit:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.repository_audit .
```

CI validates Python syntax, repository shape, provider configuration, memory behavior, and the chat CLI. A passing test suite is evidence for those tested behaviors only; it is not a claim that TARA is generally intelligent or production-ready.
