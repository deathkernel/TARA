"""Local TARA chat API for the in-repository React frontend."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .brain import TARABrain
from .env import load_dotenv
from .ollama_provider import OllamaProvider

MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
MAX_CONTEXT_CHARS = 16000


class ChatRequestError(ValueError):
    """A user-supplied chat payload is invalid."""


def build_conversation_prompt(messages: Any) -> str:
    """Validate a bounded transcript and format it for TARA's brain."""
    if not isinstance(messages, list) or not messages:
        raise ChatRequestError("Send at least one chat message.")
    if len(messages) > MAX_MESSAGES:
        raise ChatRequestError(f"Too many messages (maximum {MAX_MESSAGES}). Start a new chat.")

    transcript: list[str] = []
    total_chars = 0
    for message in messages:
        if not isinstance(message, dict):
            raise ChatRequestError("Each message must be an object.")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise ChatRequestError("Message role must be 'user' or 'assistant'.")
        if not isinstance(content, str) or not content.strip():
            raise ChatRequestError("Message content cannot be empty.")
        content = content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            raise ChatRequestError(f"Each message must be {MAX_MESSAGE_CHARS} characters or fewer.")
        total_chars += len(content)
        if total_chars > MAX_CONTEXT_CHARS:
            raise ChatRequestError("Conversation context is too long. Start a new chat.")
        transcript.append(f"{'User' if role == 'user' else 'TARA'}: {content}")

    if messages[-1].get("role") != "user":
        raise ChatRequestError("The last message must be from the user.")
    return (
        "You are TARA in a local browser chat. Use the transcript for context. "
        "Answer the latest user message, do not repeat the transcript, and do not claim "
        "actions that are not evidenced.\n\n"
        + "\n".join(transcript)
        + "\n\nTARA:"
    )


def answer_chat(tara: TARABrain, payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise ChatRequestError("Request body must be a JSON object.")
    prompt = build_conversation_prompt(payload.get("messages"))
    response = tara.respond(prompt, max_new_tokens=512, temperature=0.6)
    text = getattr(response, "text", None)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("TARA returned an empty response.")
    return {"reply": text.strip()}


def get_health(provider: OllamaProvider) -> dict[str, Any]:
    """Check Ollama connectivity and whether the configured model is installed."""
    try:
        with urlopen(f"{provider.base_url}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Ollama returned an unexpected model-list response.")
        models = data.get("models", [])
        names = [item.get("name", "") for item in models if isinstance(item, dict) and isinstance(item.get("name", ""), str)]
        installed = provider.model in names
        return {
            "ok": installed, "provider": "Ollama (local)", "ollama": "connected",
            "model": provider.model, "model_installed": installed, "available_models": names,
            "message": "Ready" if installed else f"Model '{provider.model}' is not installed. Run: ollama pull {provider.model}",
        }
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "ok": False, "provider": "Ollama (local)", "ollama": "disconnected",
            "model": provider.model, "model_installed": False, "available_models": [],
            "message": f"Cannot verify Ollama at {provider.base_url}. Open Ollama and retry. ({exc})",
        }


def make_handler(tara: TARABrain, provider: OllamaProvider):
    class TARARequestHandler(BaseHTTPRequestHandler):
        server_version = "TARAChat/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/api/health":
                self._send_json(200, get_health(provider))
                return
            self._send_json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json(404, {"error": "Not found."})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except (TypeError, ValueError):
                self._send_json(411, {"error": "A valid Content-Length header is required."})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Request must be between 1 and {MAX_BODY_BYTES} bytes."})
                return
            try:
                raw = self.rfile.read(length).decode("utf-8")
                result = answer_chat(tara, json.loads(raw))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid UTF-8 JSON."})
                return
            except ChatRequestError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except (RuntimeError, TimeoutError, URLError) as exc:
                self._send_json(503, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"error": f"TARA could not answer ({type(exc).__name__}). Check the server terminal."})
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args: Any) -> None:
            # Do not log chat message bodies.
            super().log_message(format, *args)

    return TARARequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TARA's local browser chat API.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (local-only by default).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000).")
    parser.add_argument("--model", default=None, help="Ollama model (defaults to TARA_LLAMA_MODEL or llama3.2:3b).")
    parser.add_argument("--ollama-host", default=None, help="Ollama URL (defaults to OLLAMA_HOST or localhost:11434).")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    provider = OllamaProvider(model=args.model, base_url=args.ollama_host)
    tara = TARABrain(model=None, llm_provider=provider)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tara, provider))
    print(f"TARA chat API: http://{args.host}:{args.port}")
    print(f"Ollama model: {provider.model}")
    print("Start the React UI with: cd frontend && npm run dev")
    print("This API binds to localhost by default. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTARA chat API stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
