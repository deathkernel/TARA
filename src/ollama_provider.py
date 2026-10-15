"""Local Llama inference through the Ollama HTTP API.

TARA keeps memory, planning, tools, and verification. Ollama serves a
pretrained local language model; this adapter does not train or fine-tune it.
The implementation uses Python's standard library and is safe to test with an
injected transport, without a running Ollama server or downloaded model.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


JsonTransport = Callable[[str, dict[str, Any], float], dict[str, Any]]


def _http_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Could not reach the local Ollama server at {url}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Ollama returned an invalid JSON response")
    return data


class OllamaProvider:
    """TARA-compatible local LLM provider backed by an Ollama server."""

    def __init__(
        self,
        *,
        model: str = "llama3.2:3b",
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        transport: JsonTransport | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model.strip()
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self._transport = transport or _http_json

    @classmethod
    def from_env(cls) -> "OllamaProvider":
        """Build the provider from local Ollama/TARA environment settings."""
        raw_timeout = os.getenv("TARA_LLM_TIMEOUT_SECONDS", "120").strip()
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise ValueError("TARA_LLM_TIMEOUT_SECONDS must be a number") from exc
        return cls(
            model=os.getenv("TARA_LLAMA_MODEL", "llama3.2:3b").strip(),
            base_url=os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").strip(),
            timeout=timeout,
        )

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_completion_tokens: int = 1024,
        temperature: float = 0.6,
    ) -> str:
        """Generate one response locally; no API key or cloud request is used."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")

        messages: list[dict[str, str]] = []
        if system and system.strip():
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_completion_tokens,
            },
        }
        result = self._transport(f"{self.base_url}/api/chat", payload, self.timeout)
        if result.get("error"):
            raise RuntimeError(f"Ollama error: {result['error']}")
        message = result.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("Ollama response did not contain message.content")
        return message["content"]


__all__ = ["OllamaProvider"]
