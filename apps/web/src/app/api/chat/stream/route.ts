import { NextRequest, NextResponse } from "next/server";
import { jsonError, readJsonObject, validateMessage } from "@/lib/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// Deliberately mocked SSE endpoint. It does not call an LLM or persist messages.
export async function POST(request: NextRequest) {
  try {
    const body = await readJsonObject(request);
    const message = validateMessage(body.message);
    const reply = `This is TARA's local streaming preview. I received: "${message}". Once the Python inference service is connected, this mock response will be replaced with a real model response.`;
    const tokens = reply.match(/\S+\s*/g) ?? [reply];
    const encoder = new TextEncoder();

    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          for (const token of tokens) {
            controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "token", token })}\n\n`));
            await new Promise((resolve) => setTimeout(resolve, 18));
          }
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({ type: "done" })}\n\n`));
        } finally {
          controller.close();
        }
      },
    });

    return new Response(stream, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    return jsonError(error);
  }
}
