"""Model routing and capability-aware selection for TARA."""

from __future__ import annotations

from dataclasses import dataclass
from .model_backends import MODEL_REGISTRY, TransformersBackend, create_backend


@dataclass(frozen=True)
class RouteDecision:
    backend: str
    reason: str


class ModelRouter:
    """Small deterministic router; policy can later be replaced by a learned router."""

    def __init__(self):
        self._backends: dict[str, TransformersBackend] = {}

    def choose(self, prompt: str, *, preferred: str | None = None) -> RouteDecision:
        if preferred:
            key = preferred.lower()
            if key == "native":
                return RouteDecision("native", "explicit native backend")
            if key in MODEL_REGISTRY:
                return RouteDecision(key, "explicit external backend")
            raise ValueError(f"unknown model backend: {preferred}")

        text = prompt.lower()
        if any(word in text for word in ("python", "code", "program", "function", "debug")):
            return RouteDecision("codellama", "coding-oriented prompt")
        if any(word in text for word in ("math", "calculate", "equation", "sum", "multiply")):
            return RouteDecision("native", "small deterministic task")
        return RouteDecision("llama", "general language fallback")

    def get(self, backend: str) -> TransformersBackend:
        if backend not in self._backends:
            self._backends[backend] = create_backend(backend)
        return self._backends[backend]

    def clear(self) -> None:
        self._backends.clear()
