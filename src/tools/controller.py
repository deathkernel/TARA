"""Bounded tool selection, argument validation, execution verification and rollback."""
from __future__ import annotations

from dataclasses import dataclass
import inspect
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


@dataclass(frozen=True)
class ToolSpec:
    function: Callable[..., Any]
    validator: Callable[[dict[str, Any]], tuple[bool, str]] | None = None
    requires_confirmation: bool = False


class ToolController:
    """Execute only registered tools with explicit argument/policy checks."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._stopped = False
        self._rollback: list[Callable[[], Any]] = []

    def register(self, name: str, function: Callable[..., Any], *, validator=None, requires_confirmation=False) -> None:
        if not name.strip():
            raise ValueError("tool name must not be empty")
        if not callable(function):
            raise TypeError("tool function must be callable")
        if validator is not None and not callable(validator):
            raise TypeError("tool validator must be callable")
        self._tools[name] = ToolSpec(function, validator, bool(requires_confirmation))

    def stop(self) -> None:
        self._stopped = True

    def resume(self) -> None:
        self._stopped = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    def decide(self, name: str, *, confirmed: bool = False) -> ToolDecision:
        if self._stopped:
            return ToolDecision(name, False, "emergency stop is active")
        spec = self._tools.get(name)
        if spec is None:
            return ToolDecision(name, False, "tool is not registered")
        if spec.requires_confirmation and not confirmed:
            return ToolDecision(name, False, "tool requires explicit confirmation")
        return ToolDecision(name, True, "tool is registered and policy checks passed")

    @staticmethod
    def _validate_signature(function, kwargs: dict[str, Any]) -> tuple[bool, str]:
        try:
            inspect.signature(function).bind(**kwargs)
        except TypeError as exc:
            return False, f"invalid tool arguments: {exc}"
        return True, ""

    def execute(self, name: str, *, confirmed: bool = False, **kwargs: Any) -> ToolExecution:
        decision = self.decide(name, confirmed=confirmed)
        if not decision.allowed:
            return ToolExecution(decision, error=decision.reason)
        spec = self._tools[name]
        valid, reason = self._validate_signature(spec.function, kwargs)
        if not valid:
            return ToolExecution(decision, error=reason)
        if spec.validator is not None:
            try:
                valid, reason = spec.validator(dict(kwargs))
            except Exception as exc:
                return ToolExecution(decision, error=f"validator failure: {type(exc).__name__}: {exc}")
            if not valid:
                return ToolExecution(decision, error=reason or "tool arguments rejected by policy")
        try:
            output = spec.function(**kwargs)
            return ToolExecution(decision, output=output, success=True)
        except Exception as exc:
            return ToolExecution(decision, error=f"{type(exc).__name__}: {exc}")

    def execute_with_rollback(self, name: str, *, rollback=None, confirmed=False, **kwargs: Any) -> ToolExecution:
        if rollback is not None:
            self._rollback.append(rollback)
        result = self.execute(name, confirmed=confirmed, **kwargs)
        if not result.success and rollback is not None:
            try:
                rollback()
            except Exception:
                pass
        return result

    def rollback_last(self) -> bool:
        if not self._rollback:
            return False
        action = self._rollback.pop()
        try:
            action()
            return True
        except Exception:
            return False
