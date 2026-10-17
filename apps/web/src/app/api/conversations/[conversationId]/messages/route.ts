import { NextRequest, NextResponse } from "next/server";
import { jsonError, readJsonObject, requireUuid, validateMessage, withUserTransaction } from "@/lib/api";

export const runtime = "nodejs";
type RouteContext = { params: Promise<{ conversationId: string }> };

export async function GET(request: NextRequest, context: RouteContext) {
  try {
    const { conversationId: rawId } = await context.params;
    const id = requireUuid(rawId);
    const messages = await withUserTransaction(request, async (client) => {
      const ownerCheck = await client.query("SELECT id FROM conversations WHERE id = $1", [id]);
      if (ownerCheck.rowCount !== 1) return null;
      const result = await client.query(
        `SELECT id, conversation_id, role, content, created_at
           FROM messages WHERE conversation_id = $1
          ORDER BY created_at ASC, id ASC LIMIT 500`,
        [id],
      );
      return result.rows;
    });
    if (!messages) return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
    return NextResponse.json({ messages });
  } catch (error) {
    return jsonError(error);
  }
}

export async function POST(request: NextRequest, context: RouteContext) {
  try {
    const { conversationId: rawId } = await context.params;
    const id = requireUuid(rawId);
    const body = await readJsonObject(request);
    const content = validateMessage(body.content);
    const message = await withUserTransaction(request, async (client) => {
      const ownerCheck = await client.query("SELECT id FROM conversations WHERE id = $1", [id]);
      if (ownerCheck.rowCount !== 1) return null;
      const result = await client.query(
        `INSERT INTO messages (conversation_id, role, content)
         VALUES ($1, 'user', $2)
         RETURNING id, conversation_id, role, content, created_at`,
        [id, content],
      );
      await client.query("UPDATE conversations SET updated_at = now() WHERE id = $1", [id]);
      return result.rows[0];
    });
    if (!message) return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
    return NextResponse.json({ message }, { status: 201 });
  } catch (error) {
    return jsonError(error);
  }
}
