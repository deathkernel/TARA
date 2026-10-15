import { NextRequest, NextResponse } from "next/server";
import { jsonError, readJsonObject, validateTitle, withUserTransaction } from "@/lib/api";

export const runtime = "nodejs";

export async function GET(request: NextRequest) {
  try {
    const conversations = await withUserTransaction(request, async (client) => {
      const result = await client.query(
        `SELECT id, title, created_at, updated_at
           FROM conversations
          ORDER BY updated_at DESC, id DESC
          LIMIT 100`,
      );
      return result.rows;
    });
    return NextResponse.json({ conversations });
  } catch (error) {
    return jsonError(error);
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await readJsonObject(request);
    const title = body.title === undefined ? "New conversation" : validateTitle(body.title);
    const conversation = await withUserTransaction(request, async (client, userId) => {
      const result = await client.query(
        `INSERT INTO conversations (owner_id, title)
         VALUES ($1, $2)
         RETURNING id, title, created_at, updated_at`,
        [userId, title],
      );
      return result.rows[0];
    });
    return NextResponse.json({ conversation }, { status: 201 });
  } catch (error) {
    return jsonError(error);
  }
}
