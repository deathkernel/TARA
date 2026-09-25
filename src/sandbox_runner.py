"""Fail-closed runner for generated Python candidates.

TARA does not execute generated source on the user's host. Candidate validation
remains available for analysis, while execution is intentionally disabled.
Untrusted execution belongs in a dedicated container/VM service.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass


ALLOWED_MODULES = {
    "bisect", "collections", "decimal", "fractions", "functools", "heapq",
    "itertools", "json", "math", "operator", "random", "re", "statistics",
    "string",
}
BLOCKED_CALLS = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "getattr", "setattr", "delattr", "globals", "locals", "vars", "breakpoint",
}
BLOCKED_NAMES = {
    "object", "type", "super", "__builtins__", "__loader__", "__spec__",
    "__package__", "__name__",
}


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    output: object | None = None
    error: str = ""


def validate_candidate(source: str) -> tuple[bool, str]:
    """Perform conservative static validation without executing the source."""
    if not isinstance(source, str) or not source.strip():
        return False, "candidate source is empty"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not any(node.name == "solve" for node in functions):
        return False, "candidate must define solve(value)"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in ALLOWED_MODULES:
                    return False, f"import not allowed: {root}"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in ALLOWED_MODULES:
                return False, f"import not allowed: {root}"
        elif isinstance(node, ast.Name) and node.id in BLOCKED_NAMES:
            return False, f"blocked name: {node.id}"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS | BLOCKED_NAMES:
                return False, f"blocked call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("_"):
                return False, f"blocked private attribute: {node.func.attr}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return False, f"blocked private attribute: {node.attr}"

    return True, ""


def run_candidate(
    source: str,
    value: object,
    timeout: float = 2.0,
    max_output_bytes: int = 1_000_000,
) -> SandboxResult:
    """Never execute generated source on the host; return a safe failure."""
    valid, reason = validate_candidate(source)
    if not valid:
        return SandboxResult(False, error=reason)
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be positive")
    return SandboxResult(
        False,
        error="generated-code execution is disabled; configure an isolated execution backend",
    )
