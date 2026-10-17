import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import type { PoolClient } from "pg";
import { withTransaction } from "@/lib/db";

const UUID_PATTERN = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export class ApiError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
    this.name = "ApiError";
  }
}

export function isUuid(value: string): boolean {
  return UUID_PATTERN.test(value);
}

export function jsonError(error: unknown): NextResponse {
  if (error instanceof ApiError) return NextResponse.json({ error: error.message }, { status: error.status });
  if (error instanceof Error && error.message === "TARA_DATABASE_URL is not configured") {
    return NextResponse.json({ error: "Database is not configured" }, { status: 503 });
  }
  return NextResponse.json({ error: "Internal server error" }, { status: 500 });
}

export async function readJsonObject(request: NextRequest): Promise<Record<string, unknown>> {
  const rawBody = await request.text();
  if (new TextEncoder().encode(rawBody).byteLength > 16_384) throw new ApiError(413, "Request body is too large");
  let value: unknown;
  try {
    value = JSON.parse(rawBody);
  } catch {
    throw new ApiError(400, "Expected a valid JSON object");
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new ApiError(400, "Expected a JSON object");
  return value as Record<string, unknown>;
}

/**
 * Authentication is not wired yet. Production fails closed. Local development
 * can opt into one fixed identity; request fields and headers never define identity.
 */
export async function withUserTransaction<T>(
  _request: NextRequest,
  callback: (client: PoolClient, userId: string) => Promise<T>,
): Promise<T> {
  const userId = process.env.NODE_ENV === "production" ? null : process.env.TARA_DEV_USER_ID;
  if (!userId || !isUuid(userId)) throw new ApiError(401, "Authentication is not configured for this runtime");

  return withTransaction(async (client) => {
    await client.query("SELECT set_config('tara.user_id', $1, true)", [userId]);
    await client.query("INSERT INTO app_users (id) VALUES ($1) ON CONFLICT (id) DO NOTHING", [userId]);
    return callback(client, userId);
  });
}

export function validateTitle(value: unknown): string {
  if (typeof value !== "string") throw new ApiError(400, "title must be a string");
  const title = value.trim();
  if (title.length < 1 || title.length > 120) throw new ApiError(400, "title must contain between 1 and 120 characters");
  return title;
}

export function validateMessage(value: unknown): string {
  if (typeof value !== "string") throw new ApiError(400, "content must be a string");
  const content = value.trim();
  if (content.length < 1 || content.length > 12_000) throw new ApiError(400, "content must contain between 1 and 12000 characters");
  return content;
}

export function requireUuid(value: string): string {
  if (!isUuid(value)) throw new ApiError(400, "Invalid resource ID");
  return value;
}
