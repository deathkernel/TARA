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

Memory is stored locally at `data/tara_memory.json` by default. Set `TARA_MEMORY_PATH` to change the location. Writes use a temporary file and atomic replacement so a failed serialization or interrupted write does not truncate the last saved file. Loads validate the complete supported schema before replacing in-memory state; malformed or unknown-version files stop startup with an actionable error rather than silently discarding data. The original plain key/value JSON format remains readable. Saved memories are separate from the active conversation context. Do not store passwords, API keys, or other secrets in memory.

## Conversation settings

| Variable | Default | Purpose |
|---|---|---|
| `TARA_LANGUAGE` | `auto` | Preferred response language; auto follows the conversation |
| `TARA_PERSONALITY` | `balanced` | Response style: balanced, friendly, professional, concise, or a custom style description |
| `TARA_MAX_HISTORY_TURNS` | `12` (maximum `64`) | Maximum recent user/assistant turns retained in session context |
| `TARA_MAX_RESPONSE_TOKENS` | Provider profile default: `128` eco, `256` balanced, `512` performance | Maximum response length requested from the provider; explicit values must be between 1 and 4,096 |

TARA rejects user messages longer than 16,000 characters, bounds assembled provider context to 32,000 characters, and rejects provider responses longer than 32,000 characters. These character limits are safety bounds, not tokenizer-based context-window calculations. The provider's own token/context limits still apply.

## Memory retrieval

The regression suite includes a small fixture-based retrieval evaluation covering Hindi, accented Latin text, Unicode compatibility normalization, no-match behavior, and stable ranking ties. These are deterministic lexical tests, not a semantic-understanding benchmark.

Saved-memory search uses deterministic lexical overlap rather than a semantic embedding model. Matching is Unicode-aware, so scripts such as Hindi/Devanagari and accented Latin text can be searched without extra dependencies. Results are ranked by query-token overlap; ties retain memory insertion order. A zero result limit returns no records, and invalid limits or relevance thresholds are rejected. This improves multilingual token handling but does not provide semantic translation or guarantee that related words with different spellings will match.

## Provider contract

A language provider implements `complete(prompt, *, system, max_completion_tokens, temperature)` and returns plain text. It exposes `default_max_tokens` for its resource profile, applies a finite request timeout, and raises an exception for transport errors or malformed responses. Providers perform inference only; TARA owns chat history and memory. The interface is documented in `src/provider_contract.py`.

## Provider and resource settings

| Variable | Default | Purpose |
|---|---|---|
| `TARA_LLM_BACKEND` | `ollama` | Local provider selection |
| `TARA_LLAMA_MODEL` | `llama3.2:3b` | Ollama model name |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Local Ollama endpoint |
| `TARA_LLM_TIMEOUT_SECONDS` | `120` | Request timeout |
| `TARA_POWER_MODE` | `eco` | Resource profile: eco, balanced, or performance |
| `TARA_MEMORY_PATH` | `data/tara_memory.json` | Persistent memory file |

The provider validates its endpoint and resource limits. The default endpoint is local-only; do not expose Ollama to an untrusted network.

## Development and validation

Run the conversation-focused tests and static repository audit:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.repository_audit .
```

CI validates Python syntax, repository shape, provider configuration, memory behavior, and the chat CLI. A passing test suite is evidence for those tested behaviors only; it is not a claim that TARA is generally intelligent or production-ready.


### Retrieval evaluation metrics

The fixture runner in `src/memory_evaluation.py` reports precision@k, recall@k, hit rate@k, mean reciprocal rank (MRR), and no-match accuracy. Ranking metrics are calculated only for cases that have at least one expected relevant memory; no-match accuracy is reported separately for cases whose expected result is empty. This keeps the small fixture set transparent and prevents empty-result cases from inflating ranking scores. These metrics measure lexical retrieval against hand-labelled fixtures, not semantic quality across arbitrary user conversations.


### Integration readiness checks

Before proposing a release, run the full test suite and static audit on Python 3.11, 3.12, and 3.13. Persistence tests cover atomic-write failure, invalid metadata, unknown format versions, legacy-file compatibility, and startup behavior with corrupt saved memory. CLI regressions verify that `/clear` removes active context while leaving explicitly saved memory intact. A green CI run is evidence for these tested paths; it does not replace a release review on the target machine with its configured Ollama model.
