# TARA Chat Frontend

React + TypeScript + Vite chat UI for TARA. The UI is designed to call the backend API; it does not simulate assistant replies.

## Run

```bash
cd frontend
npm install
npm run dev
```

Vite serves the frontend at http://localhost:5173.

## Backend connection

By default, requests target the same origin at `/api/health` and `/api/chat`. Set `VITE_TARA_API_URL` to a backend origin when running the frontend separately, for example in a local `.env.local` file:

```dotenv
VITE_TARA_API_URL=http://localhost:8000
```

The backend is expected to expose:
- `GET /api/health`: JSON response containing `{"status":"ok","provider_configured":true|false}`. The UI validates `status` and separately displays provider configuration.
- `POST /api/chat`: JSON request `{"message":"..."}`, response `{"message":"assistant response"}`.

These endpoints are an integration contract and must be implemented/verified on the Python side. The UI shows a connection error instead of inventing a reply when the backend is unavailable. `Enter` sends a message, `Shift+Enter` inserts a newline, `Ctrl+K` / `⌘K` starts a new chat, and `Escape` closes dialogs and mobile navigation. Recent conversations (up to 30) and the selected theme are stored locally in this browser; they are not synced to an account. Use **Recent chats** in the sidebar to expand or collapse the history list.

## Build

```bash
npm run build
```

No generated build output is committed. Do not put provider API keys in `VITE_*` variables; those are exposed to the browser.


## Start the Python API

The current adapter uses the repository's existing `GroqProvider`. It does not claim to wrap every TARA cognitive module.

From the repository root:

```bash
python -m pip install -r requirements-web.txt
# Configure GROQ_API_KEY in your backend environment (never in VITE_* variables)
uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

Then start the frontend in another terminal with `npm install && npm run dev` from `frontend/`. The Vite development proxy forwards `/api` requests to `http://127.0.0.1:8000` by default; override with `TARA_API_PROXY_TARGET` if needed.

Frontend unit tests cover persisted-history validation (including blank-ID rejection), API response/error parsing and unknown thrown-value normalization, health endpoint payload validation, chat request lifecycle guards, stale/aborted health-check lifecycle decisions, keyboard shortcut decisions (including IME-safe Escape handling and bounded textarea sizing), and modal focus-trap navigation. Run them with `npm test` from `frontend/`. The Frontend CI workflow runs the unit tests and TypeScript/Vite production build on pull requests. API tests live in `tests/test_web_api.py` and require the project's test dependencies plus `fastapi` and `httpx`. Run them with `python -m pytest tests/test_web_api.py`.

The API health endpoint reports process health separately from whether a provider key is configured. Provider credentials remain server-side. Configure `TARA_CORS_ORIGINS` only for browser origins that should be allowed.
