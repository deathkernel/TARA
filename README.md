# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a modular Python reasoning application with a command-line chat interface and separable components for perception, planning, memory, tool use, reflection, and world-state context. It uses a configured language-model backend for text generation; with the local setup below, Ollama runs the model and TARA uses it for inference. Model training is not part of the normal chat workflow.

## Run locally

Install the Python dependencies, make sure Ollama is running, and install the model you want to use:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src
```

Set `TARA_LLM_BACKEND` to configure the language-model provider. See [`.env.example`](.env.example) for available environment settings. Use `/help` in chat to see the commands available in your checkout.

## Project structure

TARA keeps its main responsibilities in separate Python modules so components can be inspected and tested independently. Tool access and external integrations should remain explicitly configured and bounded; model-generated text is not itself authorization to perform an action.

## Public API commands

TARA supports explicit, read-only searches against GitHub, Wikipedia, Open-Meteo weather, Hacker News, and Stack Exchange:

```text
/api github <query>
/api wiki <query>
/api weather <city>
/api hn <query>
/api stackoverflow <query>
```

Remote results include source links and are treated as untrusted input. Requests are bounded, and remote results are not saved to persistent conversation memory or training data. See [the API guide](docs/public_api_tools.md) for details.

## Optional voice output

The project may be configured for ElevenLabs voice output. Configure credentials locally through environment variables and never commit API keys or other secrets to Git. Check the relevant source module and current SDK documentation for supported options.

## Development checks

Run the test suite with:

```bash
pytest
```

External services may require credentials and network access for live integration tests. Prefer mocked responses for deterministic unit tests.
