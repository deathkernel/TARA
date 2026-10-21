# TARA Integrated Browser Frontend

The React frontend lives in `frontend/` inside the TARA repository. The Python
adapter in `src/web_app.py` serves the production build from `frontend/dist/`
and routes chat requests through the existing `TARABrain` and configured
Ollama provider. There is no separate application repository or second chat
engine.

## Local setup

1. Install Python dependencies using the repository's existing environment:
   `python -m pip install -r requirements.txt`
2. Install Node.js 20+.
3. From `frontend/`, install dependencies and build the UI:
   `npm install` then `npm run build`
4. Ensure Ollama is running and `ollama list` includes the configured model.
5. From the repository root run `python -m src.web_app`.
6. Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The default model uses the settings supported by `OllamaProvider.from_env()`,
including `TARA_LLAMA_MODEL`, `OLLAMA_HOST`, and `TARA_POWER_MODE`. CLI
overrides are available with `--model` and `--ollama-host`.

## Frontend development

Run the Python API in one terminal with `python -m src.web_app`. In a second
terminal, run `cd frontend && npm install && npm run dev`; Vite serves the
development UI at `http://127.0.0.1:5173` and proxies `/api` to the Python
server. Build output is served by the integrated Python app.

## Notes

- Browser history is local to the current browser profile; it is not server-side
  persistence or cross-device account sync.
- The health badge is based on the backend's actual Ollama connectivity and
  configured model inventory.
- The UI does not expose provider credentials and does not imply that chat can
  perform arbitrary shell or desktop actions.
