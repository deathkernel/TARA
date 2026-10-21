"""Local HTTP API connecting the React frontend to TARA's Python brain."""
from __future__ import annotations
from functools import lru_cache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .brain import TARABrain
from .ollama_provider import OllamaProvider, OllamaUnavailable

class ChatTurn(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=12000)
    def safe_role(self) -> str:
        return "assistant" if self.role == "assistant" else "user"

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=12000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=24)

class ChatResponse(BaseModel):
    response: str
    provider: str
    model: str

provider = OllamaProvider()
app = FastAPI(title="TARA Local API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

@lru_cache(maxsize=1)
def get_brain() -> TARABrain:
    return TARABrain(model=None, llm_provider=provider)

@app.get("/api/health")
def health() -> dict[str, str]:
    available, detail = provider.health()
    return {"status": "online" if available else "offline", "provider": "Ollama (local)",
            "model": provider.model, "ollama": "available" if available else "unavailable", "detail": detail}

@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")
    history = request.history[-16:]
    transcript = "\n".join(turn.safe_role().capitalize() + ": " + turn.content.strip() for turn in history)
    prompt = ("Conversation so far:\n" + transcript + "\n\nCurrent user message:\n" + message) if transcript else message
    try:
        result = get_brain().respond(prompt, max_new_tokens=768, temperature=0.6)
    except OllamaUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail="TARA runtime could not answer: " + str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="TARA encountered an internal error.") from exc
    text = getattr(result, "text", "")
    if not isinstance(text, str) or not text.strip():
        raise HTTPException(status_code=502, detail="TARA returned an empty response.")
    return ChatResponse(response=text.strip(), provider="Ollama (local)", model=provider.model)
