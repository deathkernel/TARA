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
- `GET /api/health`: non-sensitive health response; any successful 2xx JSON response is treated as reachable.
- `POST /api/chat`: JSON request `{"message":"..."}`, response `{"message":"assistant response"}`.

These endpoints are an integration contract and must be implemented/verified on the Python side. The UI shows a connection error instead of inventing a reply when the backend is unavailable.

## Build

```bash
npm run build
```

No generated build output is committed. Do not put provider API keys in `VITE_*` variables; those are exposed to the browser.
