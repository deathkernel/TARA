"""Basic tool selection, execution verification, emergency stop and rollback."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable

@dataclass(frozen=True)
class ToolDecision:
    tool: str
    allowed: bool
    reason: str

@dataclass(frozen=True)
class ToolExecution:
    decision: ToolDecision
    output: Any = None
    success: bool = False
    error: str | None = None

class ToolController:
    def __init__(self) -> None:
        self._tools: dict[str, Callable[..., Any]] = {}
        self._stopped = False
        self._rollback: list[Callable[[], Any]] = []

    def register(self, name: str, function: Callable[..., Any]) -> None:
        if not name.strip(): raise ValueError("tool name must not be empty")
        self._tools[name] = function

    def stop(self) -> None: self._stopped = True
    def resume(self) -> None: self._stopped = False
    @property
    def stopped(self) -> bool: return self._stopped

    def decide(self, name: str) -> ToolDecision:
        if self._stopped: return ToolDecision(name, False, "emergency stop is active")
        if name not in self._tools: return ToolDecision(name, False, "tool is not registered")
        return ToolDecision(name, True, "tool is registered and execution is enabled")

    def execute(self, name: str, **kwargs: Any) -> ToolExecution:
        decision = self.decide(name)
        if not decision.allowed: return ToolExecution(decision, error=decision.reason)
        try:
            output = self._tools[name](**kwargs)
            return ToolExecution(decision, output=output, success=True)
        except Exception as exc:
            return ToolExecution(decision, error=f"{type(exc).__name__}: {exc}")

    def execute_with_rollback(self, name: str, *, rollback: Callable[[], Any] | None = None, **kwargs: Any) -> ToolExecution:
        if rollback is not None: self._rollback.append(rollback)
        result = self.execute(name, **kwargs)
        if not result.success and rollback is not None:
            try: rollback()
            except Exception: pass
        return result

    def rollback_last(self) -> bool:
        if not self._rollback: return False
        action = self._rollback.pop()
        try: action(); return True
        except Exception: return False
