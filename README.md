# TARA

**Tiny Artificial Reasoning Architecture** is a modular reasoning application with a configurable language-model provider. The default chat provider is local Ollama.

## Run

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src
```

Set `TARA_LLM_BACKEND` as needed; see [`.env.example`](.env.example) for supported settings. Use `/help` in chat to see available commands.

## Public API commands

TARA supports explicit, read-only searches against GitHub, Wikipedia, Open-Meteo weather, Hacker News, and Stack Exchange:

```text
/api github <query>
/api wiki <query>
/api weather <city>
/api hn <query>
/api stackoverflow <query>
```

Remote results include source links and are marked untrusted. Requests are allowlisted and bounded; results are not saved to persistent conversation memory or training data. See [the API guide](docs/public_api_tools.md) for details.

## Development

```bash
pytest
```

For local Ollama setup, see [the local model guide](docs/local_llama.md). CI must pass before a change is considered ready to merge.
