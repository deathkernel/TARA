"""FastAPI inference adapter for TARA's web chat.

The service only accepts authenticated internal requests and never exposes the
provider key to the browser. Readiness reflects required configuration, not a
claim that the external provider is currently reachable.
"""
from __future__ import annotations

import hmac
import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from services.inference.provider import ProviderUnavailable, provider

app = FastAPI(
    title="TARA Inference Service",
    version="0.2.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=12_000)


def _authorized(authorization: str | None) -> bool:
    expected = os.getenv("TARA_INFERENCE_API_TOKEN", "").strip()
    if not expected or not authorization:
        return False
    scheme, _, supplied = authorization.partition(" ")
    return scheme.lower() == "bearer" and bool(supplied) and hmac.compare_digest(supplied, expected)


def _sse(event: dict[str, str]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")


@app.get("/health/live", tags=["health"])
def liveness() -> dict[str, str]:
    """Return success while the HTTP process can serve requests."""
    return {"status": "ok", "service": "tara-inference"}


@app.get("/health/ready", tags=["health"])
def readiness() -> JSONResponse:
    """Readiness requires both the provider key and the internal service token."""
    if not provider.configured:
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "service": "tara-inference",
                "reason": "inference provider or internal API token is not configured",
            },
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ready", "service": "tara-inference"},
    )


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    """Compatibility health route for simple container probes."""
    return liveness()


@app.post("/v1/chat/stream", tags=["inference"])
async def chat_stream(
    body: ChatRequest,
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """Stream provider-generated tokens to a trusted server-side caller."""
    if not _authorized(authorization):
        raise HTTPException(status_code=401, detail="Unauthorized")
    message = body.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be empty")
    if not provider.configured:
        raise HTTPException(status_code=503, detail="Inference provider is not configured")

    async def events() -> AsyncIterator[bytes]:
        try:
            async for event in provider.stream(message):
                yield _sse(event)
            yield _sse({"type": "done"})
        except ProviderUnavailable as exc:
            # The detail is intentionally generic and does not include upstream bodies.
            reason = str(exc) if str(exc) in {
                "Daily inference request budget exhausted",
                "Daily inference token budget exhausted",
            } else "Inference provider is temporarily unavailable"
            yield _sse({"type": "error", "error": reason})
        except Exception:
            yield _sse({"type": "error", "error": "Inference provider is temporarily unavailable"})

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )
