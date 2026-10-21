"""Serve TARA's integrated browser chat UI and route messages through TARABrain."""
from __future__ import annotations

import argparse
import json
import os
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .brain import TARABrain
from .env import load_dotenv
from .ollama_provider import OllamaProvider

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIST = ROOT / "frontend" / "dist"
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
MAX_CONTEXT_CHARS = 16000


class ChatRequestError(ValueError):
    """Raised for invalid client payloads."""


def build_conversation_prompt(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise ChatRequestError("Send at least one message.")
    if len(messages) > MAX_MESSAGES:
        raise ChatRequestError(f"Too many messages (maximum {MAX_MESSAGES}). Start a new chat.")
    transcript: list[str] = []
    total = 0
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise ChatRequestError("Each message must have a user or assistant role.")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ChatRequestError("Message content cannot be empty.")
        content = content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            raise ChatRequestError(f"Each message must be {MAX_MESSAGE_CHARS} characters or fewer.")
        total += len(content)
        if total > MAX_CONTEXT_CHARS:
            raise ChatRequestError("Conversation is too long. Start a new chat.")
        transcript.append(f"{'User' if message['role'] == 'user' else 'TARA'}: {content}")
    if messages[-1].get("role") != "user":
        raise ChatRequestError("The last message must be from the user.")
    return (
        "You are TARA, the assistant running in the TARA software project. "
        "Use the conversation transcript for context. Answer the latest user message; "
        "do not repeat the transcript and do not claim actions that have not occurred.\n\n"
        + "\n".join(transcript) + "\n\nTARA:"
    )


def answer_chat(tara: TARABrain, payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ChatRequestError("Request body must be a JSON object.")
    prompt = build_conversation_prompt(payload.get("messages"))
    response = tara.respond(prompt, max_new_tokens=512, temperature=0.6)
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("TARA returned an empty response.")
    return {"message": text.strip()}


def get_health(provider: OllamaProvider) -> dict[str, Any]:
    base_url = getattr(provider, "base_url", getattr(provider, "host", "http://127.0.0.1:11434")).rstrip("/")
    model = provider.model
    try:
        with urlopen(f"{base_url}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        models = data.get("models", []) if isinstance(data, dict) else []
        names = [
            item.get("name", "") for item in models
            if isinstance(item, dict) and isinstance(item.get("name", ""), str)
        ]
        # Ollama tags commonly include :latest, so accept exact name or equivalent base tag.
        installed = model in names or any(name.split(":")[0] == model.split(":")[0] for name in names)
        return {
            "status": "ok" if installed else "model-missing",
            "provider_configured": True,
            "provider_reachable": True,
            "model": model,
            "model_installed": installed,
            "available_models": names,
        }
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return {
            "status": "offline",
            "provider_configured": True,
            "provider_reachable": False,
            "model": model,
            "model_installed": False,
            "available_models": [],
        }


def make_handler(tara: TARABrain, provider: OllamaProvider):
    class TARARequestHandler(BaseHTTPRequestHandler):
        server_version = "TARAWeb/1.0"

        def _json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _static(self, request_path: str) -> None:
            relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
            path = (FRONTEND_DIST / relative).resolve()
            try:
                path.relative_to(FRONTEND_DIST.resolve())
            except ValueError:
                self._json(404, {"detail": "Not found."})
                return
            if not path.is_file() and "." not in Path(relative).name:
                path = FRONTEND_DIST / "index.html"
            try:
                body = path.read_bytes()
            except OSError:
                self._json(503, {
                    "detail": "Frontend build not found. Run npm install and npm run build in the frontend folder."
                })
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if content_type in {"text/html", "text/css", "application/javascript", "text/plain"}:
                content_type += "; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache" if path.name == "index.html" else "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self._json(200, get_health(provider))
                return
            if self.path.startswith("/api/"):
                self._json(404, {"detail": "Not found."})
                return
            self._static(self.path.split("?", 1)[0])

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._json(404, {"detail": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(411, {"detail": "A valid Content-Length header is required."})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._json(413, {"detail": f"Request must be between 1 and {MAX_BODY_BYTES} bytes."})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = answer_chat(tara, payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._json(400, {"detail": "Request body must be valid UTF-8 JSON."})
                return
            except ChatRequestError as exc:
                self._json(400, {"detail": str(exc)})
                return
            except (RuntimeError, TimeoutError) as exc:
                self._json(503, {"detail": str(exc)})
                return
            except Exception as exc:
                # Keep stack traces and machine details on the server, not in browser responses.
                self._json(500, {"detail": f"TARA could not answer ({type(exc).__name__}). Check the server terminal."})
                return
            self._json(200, result)

        def log_message(self, format: str, *args: Any) -> None:
            # Standard HTTP access logs omit request bodies and therefore chat content.
            super().log_message(format, *args)

    return TARARequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TARA's integrated browser frontend.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: local-only).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000).")
    parser.add_argument("--model", default=None, help="Installed Ollama model name.")
    parser.add_argument("--ollama-host", default=None, help="Ollama URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    # Use the provider's environment-aware factory when no CLI override was supplied.
    if args.model is None and args.ollama_host is None:
        provider = OllamaProvider.from_env()
    else:
        provider = OllamaProvider.from_env()
        if args.model:
            provider.model = args.model
        if args.ollama_host:
            if hasattr(provider, "base_url"):
                provider.base_url = args.ollama_host.rstrip("/")
            else:
                provider.host = args.ollama_host.rstrip("/")
    tara = TARABrain(model=None, llm_provider=provider)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tara, provider))
    print(f"TARA integrated frontend: http://{args.host}:{args.port}")
    print(f"Ollama model: {provider.model}")
    print("Local-only by default. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTARA frontend stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
