# TARA Web

Next.js App Router frontend for the TARA monorepo. It includes a responsive chat UI and a Server-Sent Events (SSE) endpoint that proxies provider-backed responses through the FastAPI inference service.

## Requirements and run

- Node.js 20.9+
- npm 10+

From the repository root:

```bash
npm install
npm run web:dev
```

Open http://localhost:3000.

For provider-backed chat, run the inference service and configure `TARA_INFERENCE_URL`, `TARA_INFERENCE_API_TOKEN`, and `GROQ_API_KEY` in server environments. Keep the provider key and internal token server-only. For local database API development, configure `TARA_DATABASE_URL` and a fixed `TARA_DEV_USER_ID`; the development identity is disabled in production. See [inference service notes](../../services/inference/README.md), [server helper notes](src/lib/README.md), and [database/RLS notes](../../db/README.md).

## Checks

```bash
npm run web:lint
npm run web:typecheck
npm run web:build
```

## Current boundaries

- `apps/web`: Next.js, React, TypeScript UI and web-facing routes.
- `POST /api/chat/stream`: proxies authenticated server-to-server SSE to the configured Groq inference service; this chat UI still does not persist messages.
- `/api/conversations`: PostgreSQL-backed endpoints; production remains disabled until a trusted authentication adapter exists.
- `services/inference`: authenticated FastAPI adapter for provider-backed token streaming.
- `db/migrations`: PostgreSQL schema and row-level ownership policies.
- `rust-tokenizer`: remains independent until a measured integration contract justifies coupling.
