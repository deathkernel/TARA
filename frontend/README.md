# TARA Frontend (frontend-only)

This React + TypeScript interface lives inside the TARA repository and calls the existing local TARA HTTP endpoints. It does not add or modify Python/backend code.

## Run on Windows

1. Start the existing TARA web backend from the repository root: `python -m src.web_app`.
2. Ensure Ollama is running and the configured model is installed (`ollama list`; install if needed with `ollama pull llama3.2:3b`).
3. In a second terminal, run:

   ```bat
   cd frontend
   npm install
   npm run dev
   ```

4. Open http://127.0.0.1:5173.

Vite proxies `/api/health` and `/api/chat` to the existing backend at `http://127.0.0.1:8000`. If your backend uses another local port, set `TARA_API_PROXY_TARGET` before starting Vite.

## Existing API contract

- `GET /api/health` returns the backend's `ok`, `model`, `ollama`, and `message` fields.
- `POST /api/chat` accepts `{"messages":[{"role":"user|assistant","content":"..."}]}` and returns `{"reply":"..."}`.

Conversation history is stored in this browser's local storage. This PR changes frontend files only. It runs the React app through Vite for development; serving the compiled `dist/` folder from the Python server is intentionally outside this frontend-only change.
