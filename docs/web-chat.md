# TARA Frontend

The React + TypeScript chat UI lives inside this repository at `frontend/`. It calls the Python API in `src/web_app.py`, which uses TARA's existing `TARABrain` and local Ollama provider.

## Windows quick start

Use two terminals from the repository root.

**Terminal 1 — TARA API**

```bat
python -m src.web_app
```

**Terminal 2 — React UI**

```bat
cd frontend
npm install
npm test
npm run build
npm run dev
```

Open http://127.0.0.1:5173.

The UI proxies `/api/health` and `/api/chat` to `http://127.0.0.1:8000`. The backend binds to `127.0.0.1` by default and is not exposed to the network unless you explicitly change the host.

The model resolves from `--model`, `TARA_LLAMA_MODEL`, `OLLAMA_MODEL`, then `llama3.2:3b`. The Ollama URL resolves from `--ollama-host`, `OLLAMA_HOST`, then `http://127.0.0.1:11434`. Install the default model if needed with `ollama pull llama3.2:3b`.

Conversation history is kept in browser local storage. API requests are bounded by message count and character limits.
