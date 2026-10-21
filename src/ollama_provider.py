"""Local Ollama chat provider for TARA."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OllamaProviderError(Exception):
    """Raised when Ollama cannot return a usable chat response."""


class OllamaProvider:
    """Minimal Ollama /api/chat adapter compatible with TARABrain."""

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 180):
        self.model = (model or os.getenv("OLLAMA_MODEL") or "llama3.2:3b").strip()
        self.host = (host or os.getenv("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.timeout = timeout
        if not self.model:
            raise ValueError("Ollama model name must not be empty")
        if not self.host:
            raise ValueError("Ollama host must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_completion_tokens: int = 256,
        temperature: float = 0.6,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if max_completion_tokens <= 0:
            raise ValueError("max_completion_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": max_completion_tokens,
                "temperature": temperature,
            },
        }
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f": {details}" if details else ""
            raise OllamaProviderError(
                f"Ollama returned HTTP {exc.code}{suffix}. "
                f"Check that model '{self.model}' is installed (ollama list)."
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise OllamaProviderError(
                f"Cannot connect to Ollama at {self.host}. "
                "Start Ollama and verify the host, then retry. "
                f"Details: {exc}"
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OllamaProviderError("Ollama returned invalid JSON.") from exc

        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise OllamaProviderError(
                "Ollama response did not contain message.content."
            ) from exc
        if not isinstance(content, str):
            raise OllamaProviderError("Ollama response content was not text.")
        return content
