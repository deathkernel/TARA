import { NextRequest, NextResponse } from "next/server";
import { jsonError, readJsonObject, requireUuid, validateTitle, withUserTransaction } from "@/lib/api";

export const runtime = "nodejs";
type RouteContext = { params: Promise<{ conversationId: string }> };

export async function GET(request: NextRequest, context: RouteContext) {
  try {
    const { conversationId: rawId } = await context.params;
    const id = requireUuid(rawId);
    const conversation = await withUserTransaction(request, async (client) => {
      const result = await client.query(
        `SELECT id, title, created_at, updated_at FROM conversations WHERE id = $1`,
        [id],
      );
      return result.rows[0] ?? null;
    });
    if (!conversation) return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
    return NextResponse.json({ conversation });
  } catch (error) {
    return jsonError(error);
  }
}

export async function PATCH(request: NextRequest, context: RouteContext) {
  try {
    const { conversationId: rawId } = await context.params;
    const id = requireUuid(rawId);
    const body = await readJsonObject(request);
    const title = validateTitle(body.title);
    const conversation = await withUserTransaction(request, async (client) => {
      const result = await client.query(
        `UPDATE conversations SET title = $2, updated_at = now()
          WHERE id = $1 RETURNING id, title, created_at, updated_at`,
        [id, title],
      );
      return result.rows[0] ?? null;
    });
    if (!conversation) return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
    return NextResponse.json({ conversation });
  } catch (error) {
    return jsonError(error);
  }
}

export async function DELETE(request: NextRequest, context: RouteContext) {
  try {
    const { conversationId: rawId } = await context.params;
    const id = requireUuid(rawId);
    const deleted = await withUserTransaction(request, async (client) => {
      const result = await client.query("DELETE FROM conversations WHERE id = $1 RETURNING id", [id]);
      return result.rowCount === 1;
    });
    if (!deleted) return NextResponse.json({ error: "Conversation not found" }, { status: 404 });
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    return jsonError(error);
  }
}
