# PostgreSQL schema

The first migration creates users, conversations, messages, indexes, and row-level security (RLS) policies.

## Apply migration

Use a database owner/migration connection:

```bash
psql "$TARA_DATABASE_URL" -v ON_ERROR_STOP=1 -f db/migrations/0001_initial_chat_schema.sql
```

## Ownership model

After authenticating a request, the web backend must set `tara.user_id` from verified server-side identity for the duration of each transaction:

```sql
BEGIN;
SELECT set_config('tara.user_id', '<verified-user-uuid>', true);
-- Run parameterized application queries in this same transaction.
COMMIT;
```

Never take this UUID directly from a request body or trust a client-set session variable. The runtime database role must be a dedicated non-owner role with no `BYPASSRLS` privilege; do not run user-facing queries as a superuser or migration owner. RLS is defense in depth, not a replacement for request authentication.

## Test ownership policies

Against a disposable migrated database, run as a database administrator/superuser (the script creates a temporary role and seeds fixtures):

```bash
psql "$TARA_TEST_DATABASE_URL" -v ON_ERROR_STOP=1 -f db/tests/ownership_policies.sql
```

The script checks per-owner row visibility, a same-owner message insert, rejection of a cross-owner insert, and prevention of cross-owner updates. All test fixtures and the temporary test role are rolled back. Never point this test at production.
