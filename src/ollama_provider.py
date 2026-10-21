"""Local Ollama inference adapter used by the TARA web API."""
from __future__ import annotations
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

class OllamaUnavailable(Exception):
    """Raised when the configured local Ollama service cannot answer."""

class OllamaProvider:
    def __init__(self, *, base_url: str | None = None, model: str | None = None, timeout: float = 120.0):
        self.base_url = (base_url or os.getenv("TARA_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        self.model = model or os.getenv("TARA_OLLAMA_MODEL", "llama3.2:3b")
        self.timeout = timeout

    def complete(self, prompt: str, *, system: str | None = None, max_completion_tokens: int | None = None, temperature: float = 0.6) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages, "stream": False, "options": {
            "num_predict": max(1, min(int(max_completion_tokens or 768), 2048)),
            "temperature": max(0.05, min(float(temperature), 1.5))
        }}
        request = Request(self.base_url + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read(2_000_000).decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            raise OllamaUnavailable("Ollama returned HTTP " + str(exc.code) + ": " + detail) from exc
        except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise OllamaUnavailable("Cannot reach Ollama at " + self.base_url + ". Start Ollama and install model '" + self.model + "'.") from exc
        text = data.get("message", {}).get("content", "")
        if not isinstance(text, str) or not text.strip():
            raise OllamaUnavailable("Ollama returned an empty response.")
        return text.strip()

    def health(self) -> tuple[bool, str]:
        request = Request(self.base_url + "/api/tags", method="GET")
        try:
            with urlopen(request, timeout=3) as response:
                data = json.loads(response.read(500_000).decode("utf-8"))
            models = [item.get("name", "") for item in data.get("models", [])]
            if not any(name == self.model or name.startswith(self.model + ":") for name in models):
                return False, "Model '" + self.model + "' is not installed. Run: ollama pull " + self.model
            return True, "Ollama and configured model are available."
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False, "Ollama is not responding at " + self.base_url + "."
