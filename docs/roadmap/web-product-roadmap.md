# TARA Web Product Roadmap

This product roadmap supplements the existing engineering phase ledger in the root README; it does not replace or renumber historical phases.

- [x] 1. Create the monorepo workspace and initialize Next.js with TypeScript.
- [x] 2. Build the responsive chat layout and sidebar (UI-only preview; no persistence or model calls yet).
- [x] 3. Create the initial PostgreSQL migration and ownership-policy integration test.
- [x] 4. Implement ownership-scoped conversation CRUD and user-message endpoints (production remains disabled until trusted authentication is wired).
- [ ] 5. Build the SSE chat interface with a mocked token stream.
- [ ] 6. Create the Python inference service with a health endpoint.
- [ ] 7. Connect the web backend to the inference adapter and run an end-to-end test.

## Architecture boundary

- `apps/web`: Next.js App Router, React, TypeScript, responsive chat experience.
- Existing root Python source: preserved; later exposed through an explicit inference service adapter.
- `db/migrations`: PostgreSQL schema and row-level ownership policies.
- `rust-tokenizer`: remains independent until a measured integration contract justifies coupling.

## Security notes

- Authentication is not implemented in this batch.
- Database policies require a trusted server to set transaction-local `tara.user_id` after authentication.
- Use a dedicated non-owner runtime database role without `BYPASSRLS`.
- Never expose database credentials or trust a user ID supplied by the browser.
