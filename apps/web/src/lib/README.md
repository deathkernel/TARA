# Web server helpers

- `db.ts` provides a process-wide PostgreSQL pool and explicit transaction helper.
- `api.ts` centralizes bounded JSON parsing, input validation, safe errors, and transaction-local RLS context.

Authentication is not implemented in this milestone. Production API requests intentionally return HTTP 401 until a trusted server-side authentication resolver is connected. In non-production environments only, setting a valid `TARA_DEV_USER_ID` enables one fixed local identity. The API never trusts user IDs supplied in request JSON or headers.

Set `TARA_DATABASE_URL` to a PostgreSQL connection string for a dedicated non-owner runtime role without `BYPASSRLS`. Do not use the migration owner or a superuser for application requests.
