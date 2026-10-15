# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a modular reasoning application with structured planning, perception, tool selection, reflection, world-state tracking, and a configurable language-model provider. The default chat provider is local Ollama; the configured provider supplies the natural-language generation.

## Current scope

The repository has removed dataset acquisition and loading, training pipelines and experiments, checkpoint save/load and checkpoint-evaluation commands, and persistent conversation memory. No model checkpoint or conversation-memory data files were tracked in the Git tree when this change was prepared.

TARA's runtime does not save conversation content for later turns. A small in-process working buffer may be used by components, but it is discarded when the process ends. Training or model-weight updates do not occur during chat.

## Chat

Install the requirements, make sure Ollama is running and the desired model is installed, then start the CLI:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src
```

Set `TARA_LLM_BACKEND` to configure the provider. See `.env.example` for supported environment settings. Chat commands are `/help`, `/emotion`, and `/exit`; persistent-memory commands have been removed.

## Read-only public API tools

The optional `src.public_api_tools.PublicAPITools` client exposes curated read-only queries to the GitHub repository search API, Wikipedia OpenSearch API, and Open-Meteo geocoding/forecast APIs:

```python
from src.public_api_tools import PublicAPITools

apis = PublicAPITools()
repos = apis.search_github_repositories("open source reasoning agents")
articles = apis.search_wikipedia("artificial intelligence")
weather = apis.get_weather("Pune")
```

Results include a source URL and `trusted=False`. Remote content is treated as untrusted input and is not automatically persisted by the chat runtime. The client uses an HTTPS host allowlist, request timeouts, response-size limits, per-host pacing, JSON validation, and explicit rate-limit errors.

## Development checks

Run the test suite with:

```bash
pytest
```

External API calls are not required by the public API unit tests; they use mocked responses. CI must pass before a change is considered ready to merge.
