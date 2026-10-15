# TARA Web

Next.js App Router frontend for the TARA monorepo. The first screen is a responsive UI foundation with local sample conversations. Persistence, authentication, SSE, and inference are not wired yet.

## Requirements and run

- Node.js 20.9+
- npm 10+

From the repository root:

```bash
npm install
npm run web:dev
```

Open http://localhost:3000.

## Checks

```bash
npm run web:lint
npm run web:typecheck
npm run web:build
```

## Boundaries

- `apps/web`: Next.js, React, TypeScript UI and future web routes.
- Existing root Python package: preserved while an inference adapter is designed.
- `db/migrations`: PostgreSQL schema and ownership enforcement.
- `rust-tokenizer`: remains independent until a measured integration contract justifies coupling.
