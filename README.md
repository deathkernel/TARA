# TARA

**TARA — Tiny Artificial Reasoning Architecture**

TARA is a modular Python reasoning application with a command-line chat interface and separable components for perception, planning, memory, tool use, reflection, and world-state context. It can be configured to use a supported language-model backend.

## Run locally

Install the Python dependencies, make sure Ollama is running, and install the model you want to use:

```bash
python -m pip install -r requirements.txt
ollama pull llama3.2:3b
python -m src
```

Set `TARA_LLM_BACKEND` to configure the language-model provider. See [`.env.example`](.env.example) for available environment settings. Use `/help` in the chat interface to see the commands available in your checkout.

## Project structure

TARA keeps its main responsibilities in separate Python modules so components can be inspected and tested independently. Tool access and external integrations should remain explicitly configured and bounded; model-generated text is not itself authorization to perform an action.

## Optional voice output

The project may be configured for ElevenLabs voice output. Configure credentials locally through environment variables and never commit API keys or other secrets to Git. Check the relevant source module and current SDK documentation for supported options.

## Development checks

Run the test suite with:

```bash
pytest
```

External services may require credentials and network access for live integration tests. Prefer mocked responses for deterministic unit tests.
