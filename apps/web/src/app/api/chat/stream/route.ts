import { NextRequest } from "next/server";
import { jsonError, readJsonObject, validateMessage } from "@/lib/api";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
  try {
    const body = await readJsonObject(request);
    const message = validateMessage(body.message);
    const token = process.env.TARA_INFERENCE_API_TOKEN?.trim();
    if (!token) {
      return Response.json({ error: "Inference service is not configured" }, { status: 503 });
    }

    const configuredBase = process.env.TARA_INFERENCE_URL?.trim() || "http://127.0.0.1:8000";
    let endpoint: URL;
    try {
      endpoint = new URL("/v1/chat/stream", configuredBase);
    } catch {
      return Response.json({ error: "Inference service URL is invalid" }, { status: 503 });
    }
    if (endpoint.protocol !== "http:" && endpoint.protocol !== "https:") {
      return Response.json({ error: "Inference service URL must use HTTP or HTTPS" }, { status: 503 });
    }

    let upstream: Response;
    try {
      upstream = await fetch(endpoint, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify({ message }),
        signal: request.signal,
        cache: "no-store",
      });
    } catch {
      return Response.json({ error: "Inference service is unavailable" }, { status: 503 });
    }

    if (!upstream.ok || !upstream.body) {
      const status = upstream.status === 503 ? 503 : 502;
      return Response.json(
        { error: status === 503 ? "Inference service is unavailable" : "Inference request failed" },
        { status },
      );
    }

    return new Response(upstream.body, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "X-Accel-Buffering": "no",
      },
    });
  } catch (error) {
    return jsonError(error);
  }
}
