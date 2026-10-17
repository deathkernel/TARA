# TARA Web

Next.js App Router frontend for the TARA monorepo. It includes a responsive chat UI and a deliberately mocked Server-Sent Events (SSE) endpoint so the streaming interaction can be developed before real inference is connected.

## Requirements and run

- Node.js 20.9+
- npm 10+

From the repository root:

```bash
npm install
npm run web:dev
```

Open http://localhost:3000.

For local database API development, configure `TARA_DATABASE_URL` and a fixed `TARA_DEV_USER_ID` in your local environment. The development identity is disabled in production. See [server helper notes](src/lib/README.md) and [database/RLS notes](../../db/README.md).

## Checks

```bash
npm run web:lint
npm run web:typecheck
npm run web:build
```

## Current boundaries

- `apps/web`: Next.js, React, TypeScript UI and web-facing routes.
- `POST /api/chat/stream`: mock-only SSE demo; does not call an LLM or persist data.
- `/api/conversations`: PostgreSQL-backed endpoints; production remains disabled until a trusted authentication adapter exists.
- Existing root Python source: preserved; the inference HTTP service is a later roadmap milestone.
- `db/migrations`: PostgreSQL schema and row-level ownership policies.
- `rust-tokenizer`: remains independent until a measured integration contract justifies coupling.
