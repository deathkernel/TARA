"""Basic controlled keyboard/mouse interaction interface.

The host supplies the actual input backend; TARA itself only validates and
records the requested operation.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class InputResult:
    action: str
    success: bool
    output: Any = None
    error: str | None = None

class InputTools:
    def __init__(self, backend: Callable[[str, dict[str, Any]], Any] | None = None) -> None:
        self.backend = backend

    def _run(self, action: str, **kwargs: Any) -> InputResult:
        if self.backend is None: return InputResult(action, False, error="input backend is not configured")
        try: return InputResult(action, True, output=self.backend(action, kwargs))
        except Exception as exc: return InputResult(action, False, error=f"{type(exc).__name__}: {exc}")

    def key(self, key: str) -> InputResult:
        if not key.strip(): return InputResult("key", False, error="key must not be empty")
        return self._run("key", key=key)

    def type_text(self, text: str) -> InputResult:
        if not isinstance(text, str): return InputResult("type_text", False, error="text must be a string")
        return self._run("type_text", text=text)

    def click(self, x: int, y: int, *, button: str = "left") -> InputResult:
        if x < 0 or y < 0 or button not in {"left", "right", "middle"}:
            return InputResult("click", False, error="invalid click arguments")
        return self._run("click", x=x, y=y, button=button)

    def move(self, x: int, y: int) -> InputResult:
        if x < 0 or y < 0: return InputResult("move", False, error="coordinates must be non-negative")
        return self._run("move", x=x, y=y)
