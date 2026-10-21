# TARA Frontend

This React + TypeScript chat UI lives inside the TARA repository at `frontend/`. It connects to the Python API in `src/web_app.py`, which calls TARA's existing `TARABrain` and `OllamaProvider`.

## Run on Windows

Open two Command Prompt windows from the repository root.

**Terminal 1 — start the TARA API**

```bat
python -m src.web_app
```

**Terminal 2 — start the React frontend**

```bat
cd frontend
npm install
npm test
npm run build
npm run dev
```

Open http://127.0.0.1:5173.

Vite proxies `/api/health` and `/api/chat` to the local Python backend at `http://127.0.0.1:8000`. To use another backend port, set `TARA_API_PROXY_TARGET` before starting Vite.

## Ollama model

The backend resolves the model from `--model`, `TARA_LLAMA_MODEL`, `OLLAMA_MODEL`, then `llama3.2:3b`. The Ollama URL resolves from `--ollama-host`, `OLLAMA_HOST`, then `http://127.0.0.1:11434`. If needed, install the default model with `ollama pull llama3.2:3b`.

The frontend stores conversation history in browser local storage. API requests are limited by message count and character budget. The backend binds to localhost by default; change the host only if you intentionally want network access.
