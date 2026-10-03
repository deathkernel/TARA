"""Safe tool contracts for TARA Baby automation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    handler: Callable[..., Any]
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ToolCall:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    tool: str
    success: bool
    output: str
    error: str | None = None


class ToolRegistry:
    """Explicit allowlist; unknown tool names cannot execute."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if not spec.name.strip():
            raise ValueError("tool name must not be empty")
        if spec.name in self._tools:
            raise ValueError(f"tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"unknown tool: {name}") from exc

    def execute(self, call: ToolCall) -> ToolResult:
        spec = self.get(call.tool)
        if spec.requires_confirmation:
            return ToolResult(
                tool=call.tool,
                success=False,
                output="",
                error="confirmation required",
            )
        try:
            result = spec.handler(**call.arguments)
        except Exception as exc:
            return ToolResult(
                tool=call.tool,
                success=False,
                output="",
                error=f"{type(exc).__name__}: {exc}",
            )
        return ToolResult(tool=call.tool, success=True, output=str(result))
