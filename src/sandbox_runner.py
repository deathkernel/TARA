"""Defense-in-depth runner for generated Python algorithm candidates.

The candidate must define ``solve(value)``. Input/output are exchanged through
JSON. Static AST checks reject common filesystem, process, network, reflection,
and dynamic-code primitives before a subprocess is started.

This is not a perfect security boundary. Truly untrusted code should run in a
container or VM with OS-level restrictions.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
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
    """Perform conservative static validation before execution."""
    if not isinstance(source, str) or not source.strip():
        return False, "candidate source is empty"
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, f"syntax error: {exc}"

    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
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
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BLOCKED_CALLS | BLOCKED_NAMES:
                return False, f"blocked call: {node.func.id}"
            if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                return False, f"blocked dunder attribute: {node.func.attr}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"blocked dunder attribute: {node.attr}"

    return True, ""


def run_candidate(source: str, value: object, timeout: float = 2.0, max_output_bytes: int = 1_000_000) -> SandboxResult:
    """Run a validated candidate in a short-lived isolated Python process."""
    valid, reason = validate_candidate(source)
    if not valid:
        return SandboxResult(False, error=reason)
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    payload = json.dumps(value, separators=(",", ":"))
    wrapper = (
        "import json\n"
        + source
        + "\n_result = solve(json.loads(__import__('sys').stdin.read()))\n"
        + "print(json.dumps(_result, separators=(',', ':')))\n"
    )
    # __import__('sys') above is needed only by the trusted wrapper. The
    # candidate itself cannot contain that call because validate_candidate runs
    # on candidate source, not on this wrapper.
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-S", "-c", wrapper],
            input=payload,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(False, error=f"timeout after {timeout:.2f}s")
    except OSError as exc:
        return SandboxResult(False, error=f"process error: {exc}")

    if len(proc.stdout.encode("utf-8", errors="replace")) > max_output_bytes:
        return SandboxResult(False, error="candidate output exceeded limit")
    if proc.returncode != 0:
        stderr = proc.stderr.strip().replace("\n", " ")[:1000]
        return SandboxResult(False, error=f"candidate failed (exit {proc.returncode}): {stderr}")
    try:
        return SandboxResult(True, output=json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        return SandboxResult(False, error=f"invalid JSON output: {exc}")
