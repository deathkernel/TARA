import { Pool, type PoolClient } from "pg";

declare global {
  // eslint-disable-next-line no-var
  var taraPgPool: Pool | undefined;
}

export function getPool(): Pool {
  const connectionString = process.env.TARA_DATABASE_URL;
  if (!connectionString) throw new Error("TARA_DATABASE_URL is not configured");
  if (!globalThis.taraPgPool) {
    globalThis.taraPgPool = new Pool({
      connectionString,
      max: Number.parseInt(process.env.TARA_DB_POOL_SIZE ?? "10", 10),
      connectionTimeoutMillis: 5_000,
      idleTimeoutMillis: 30_000,
      application_name: "tara-web",
    });
  }
  return globalThis.taraPgPool;
}

export async function withTransaction<T>(callback: (client: PoolClient) => Promise<T>): Promise<T> {
  const client = await getPool().connect();
  try {
    await client.query("BEGIN");
    const result = await callback(client);
    await client.query("COMMIT");
    return result;
  } catch (error) {
    await client.query("ROLLBACK").catch(() => undefined);
    throw error;
  } finally {
    client.release();
  }
}
