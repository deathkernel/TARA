"""Minimal HTTP adapter for the TARA chat MVP.

This adapter exposes the existing optional GroqProvider without claiming that
all cognitive modules are automatically orchestrated into chat responses.
"""
from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="TARA API", version="0.1.0")

_origins = [
    item.strip()
    for item in os.getenv("TARA_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)


class ChatResponse(BaseModel):
    message: str


@lru_cache(maxsize=1)
def _provider():
    from src.groq_provider import GroqProvider

    model = os.getenv("TARA_GROQ_MODEL", "openai/gpt-oss-20b")
    return GroqProvider(
        model=model,
        max_completion_tokens=int(os.getenv("TARA_MAX_COMPLETION_TOKENS", "1024")),
    )


@app.get("/api/health")
def health():
    """Report API process health without exposing credentials."""
    return {"status": "ok", "provider_configured": bool(os.getenv("GROQ_API_KEY"))}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be blank.")
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Groq is not configured. Set GROQ_API_KEY in the backend environment.",
        )
    try:
        answer = _provider().complete(message, system="You are TARA, a helpful AI assistant.")
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except Exception as exc:
        # Do not return provider exceptions or potentially sensitive internals.
        raise HTTPException(status_code=502, detail="The configured language provider failed to respond.") from exc
    if not isinstance(answer, str) or not answer.strip():
        raise HTTPException(status_code=502, detail="The configured language provider returned an empty response.")
    return ChatResponse(message=answer)
