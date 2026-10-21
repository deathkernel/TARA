"""Local browser chat UI for TARA, backed by the configured Ollama model."""
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .brain import TARABrain
from .env import load_dotenv
from .ollama_provider import OllamaProvider, OllamaProviderError

WEB_ROOT = Path(__file__).resolve().parent.parent / "web"
MAX_BODY_BYTES = 64 * 1024
MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
MAX_CONTEXT_CHARS = 16000


class ChatRequestError(ValueError):
    """A user-supplied chat payload is invalid."""


def build_conversation_prompt(messages: Any) -> str:
    """Validate a short chat transcript and format it for TARABrain."""
    if not isinstance(messages, list) or not messages:
        raise ChatRequestError("Send at least one chat message.")
    if len(messages) > MAX_MESSAGES:
        raise ChatRequestError(
            f"This request contains too many messages (maximum {MAX_MESSAGES}). Start a new chat."
        )

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
            raise ChatRequestError(
                f"Each message must be {MAX_MESSAGE_CHARS} characters or fewer."
            )
        total_chars += len(content)
        if total_chars > MAX_CONTEXT_CHARS:
            raise ChatRequestError("This conversation is too long. Start a new chat.")
        label = "User" if role == "user" else "TARA"
        transcript.append(f"{label}: {content}")

    if messages[-1].get("role") != "user":
        raise ChatRequestError("The last message must be from the user.")

    return (
        "You are TARA in a browser chat. Use the conversation transcript for context. "
        "Reply to the latest user message, do not repeat the transcript, and do not "
        "claim to have performed actions that are not evidenced.\n\n"
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
    """Check whether Ollama is reachable and the configured model is installed."""
    try:
        with urlopen(f"{provider.host}/api/tags", timeout=3) as response:
            data = json.loads(response.read().decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Ollama returned an unexpected model-list response.")
        models = data.get("models", [])
        names = [
            item.get("name", "")
            for item in models
            if isinstance(item, dict) and isinstance(item.get("name", ""), str)
        ]
        installed = provider.model in names
        return {
            "ok": installed,
            "ollama": "connected",
            "model": provider.model,
            "model_installed": installed,
            "available_models": names,
            "message": "Ready" if installed else f"Model '{provider.model}' is not installed. Run: ollama pull {provider.model}",
        }
    except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "ok": False,
            "ollama": "disconnected",
            "model": provider.model,
            "model_installed": False,
            "available_models": [],
            "message": f"Cannot verify Ollama at {provider.host}. Open Ollama and retry. ({exc})",
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
            if self.path in {"/", "/index.html"}:
                try:
                    body = (WEB_ROOT / "index.html").read_bytes()
                except OSError:
                    self._send_json(500, {"error": "Chat UI file is missing. Restore web/index.html."})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/api/health":
                self._send_json(200, get_health(provider))
                return
            self._send_json(404, {"error": "Not found."})

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                self._send_json(404, {"error": "Not found."})
                return
            raw_length = self.headers.get("Content-Length", "")
            try:
                length = int(raw_length)
            except (TypeError, ValueError):
                self._send_json(411, {"error": "A valid Content-Length header is required."})
                return
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Request must be between 1 and {MAX_BODY_BYTES} bytes."})
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                result = answer_chat(tara, payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(400, {"error": "Request body must be valid UTF-8 JSON."})
                return
            except ChatRequestError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            except OllamaProviderError as exc:
                self._send_json(503, {"error": str(exc)})
                return
            except Exception as exc:
                self._send_json(500, {"error": f"TARA could not answer ({type(exc).__name__}). Check the server terminal."})
                return
            self._send_json(200, result)

        def log_message(self, format: str, *args: Any) -> None:
            # Never log user message bodies.
            super().log_message(format, *args)

    return TARARequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TARA's local browser chat.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1; local only).")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000).")
    parser.add_argument("--model", default=None, help="Installed Ollama model name.")
    parser.add_argument("--ollama-host", default=None, help="Ollama URL.")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535.")
    provider = OllamaProvider(model=args.model, host=args.ollama_host)
    tara = TARABrain(model=None, llm_provider=provider)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tara, provider))
    print(f"TARA browser chat: http://{args.host}:{args.port}")
    print(f"Ollama model: {provider.model}")
    print("This server is local-only by default. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTARA browser chat stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
