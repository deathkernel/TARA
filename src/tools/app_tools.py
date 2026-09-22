"""Basic host-injected application/browser tool boundary."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class AppToolResult:
    action: str
    success: bool
    output: Any = None
    error: str | None = None

class ApplicationTools:
    """Only executes explicitly registered application actions."""
    def __init__(self) -> None:
        self._actions: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, action: Callable[..., Any]) -> None:
        if not name.strip(): raise ValueError("action name must not be empty")
        self._actions[name] = action

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._actions))

    def execute(self, name: str, **kwargs: Any) -> AppToolResult:
        action = self._actions.get(name)
        if action is None: return AppToolResult(name, False, error="application action is not registered")
        try: return AppToolResult(name, True, output=action(**kwargs))
        except Exception as exc: return AppToolResult(name, False, error=f"{type(exc).__name__}: {exc}")

class BrowserTools(ApplicationTools):
    """Browser actions use the same explicit host-injection boundary."""
    pass
