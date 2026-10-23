# TARA — Conversation AI

TARA is a conversation-first AI application. Its current scope is text chat with a local language model, bounded multi-turn context, configurable response style, and explicit user-controlled memory.

## Scope

**Included**
- Interactive chat through `python -m src`
- Local inference through [Ollama](https://ollama.com/)
- Bounded multi-turn conversation context
- Configurable response language and personality
- Explicit persistent memory commands and a context-clear command
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

TARA defaults to the local Ollama backend. It does not download models or start Ollama for you. See the .env.example file for optional settings.

## Chat commands

- `/help` — show available commands
- `/exit` — end the chat
- `/clear` — clear active conversation context without deleting saved memories
- `/remember key=value` — explicitly save a memory
- `/memory [query]` — list memories or search them
- `/forget key` — delete a saved memory

Memory is stored locally at `data/tara_memory.json` by default. Set `TARA_MEMORY_PATH` to change the location. Saved memories are separate from the active conversation context. Do not store passwords, API keys, or other secrets in memory.

## Conversation settings

| Variable | Default | Purpose |
|---|---|---|
| `TARA_LANGUAGE` | `auto` | Preferred response language; auto follows the conversation |
| `TARA_PERSONALITY` | `balanced` | Response style: balanced, friendly, professional, concise, or a custom style description |
| `TARA_MAX_HISTORY_TURNS` | `12` | Maximum recent user/assistant turns included in context |
| `TARA_MAX_RESPONSE_TOKENS` | `128` | Maximum response length requested from the provider |

## Memory retrieval

The regression suite includes a small fixture-based retrieval evaluation covering Hindi, accented Latin text, Unicode compatibility normalization, no-match behavior, and stable ranking ties. These are deterministic lexical tests, not a semantic-understanding benchmark.

Saved-memory search uses deterministic lexical overlap rather than a semantic embedding model.
Matching is Unicode-aware, so scripts such as Hindi/Devanagari and accented Latin text can be
searched without extra dependencies. Results are ranked by query-token overlap; ties retain
memory insertion order. A zero result limit returns no records, and invalid limits or relevance
thresholds are rejected. This improves multilingual token handling but does not provide semantic
translation or guarantee that related words with different spellings will match.

The `/memory query` command returns at most five lexical matches with scores; `/memory` lists at most twenty saved entries and reports any remainder. Invalid `/remember` or `/learn` input and attempts to forget a missing key do not write to disk or invoke the language model. These CLI limits keep output bounded for interactive sessions.\n\n## Provider and resource settings

| Variable | Default | Purpose |
|---|---|---|
| `TARA_LLM_BACKEND` | `ollama` | Local provider selection |
| `TARA_LLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `TARA_LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `TARA_POWER_MODE` | `eco` | Resource profile: eco, balanced, or performance |
| `TARA_MEMORY_PATH` | `data/tara_memory.json` | Persistent memory file |

Memory persistence writes to a temporary file in the destination directory and atomically replaces the saved file. Versioned memory payloads are validated before replacing in-memory state; the original plain key/value JSON format remains supported for migration. Explicit `/remember` and `/forget` mutations are rolled back in the active session if the updated snapshot cannot be saved, keeping the live memory and persisted file consistent.

The provider validates its endpoint and resource limits. The default endpoint is local-only; do not expose Ollama to an untrusted network.

## Development and validation

Run the conversation-focused tests and static repository audit:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.repository_audit .
```

CI validates Python syntax, repository shape, provider configuration, memory behavior, and the chat CLI. Provider-independent conversation contract tests use deterministic fake providers, so they do not require Ollama or model downloads. These tests check input/output normalization, history safety, explicit-memory boundaries, generation parameters, and CLI error handling; they are not a benchmark of model intelligence. Persistence tests cover legacy migration, malformed and inconsistent snapshots, state preservation after rejected loads, and CLI memory changes across reloads. A passing test suite is evidence for those tested behaviors only, not a claim that TARA is generally intelligent or production-ready.


TARA keeps retrieval deterministic: lexical overlap scores rank matching memories, equal scores retain insertion order, and successful retrieval increments access metadata. Saving the memory snapshot persists that usage metadata for later forgetting decisions.
