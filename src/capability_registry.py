"""Evidence-aware source capability inventory for TARA.

This module checks repository files only. It does not import project code, load
models, contact services, or claim that a source-level feature is active at
runtime.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    area: str
    status: str
    description: str
    present_files: tuple[str, ...]
    missing_files: tuple[str, ...]


# Keep this catalog deliberately conservative. Presence of source files is not
# evidence that a feature is configured, wired into the current runtime, or
# production-ready.
CAPABILITY_CATALOG = (
    (
        "core_reasoning",
        "reasoning",
        ("src/brain.py", "src/structured_reasoning.py"),
        "Core brain and structured reasoning source files.",
    ),
    (
        "planning",
        "orchestration",
        ("src/advanced_planning.py", "src/goal_progress.py"),
        "Planning and goal-progress source files.",
    ),
    (
        "memory",
        "memory",
        ("src/memory.py",),
        "Memory source file; persistence and runtime use require separate verification.",
    ),
    (
        "tool_framework",
        "tools",
        ("src/tools/controller.py", "src/tools/app_tools.py"),
        "Tool controller and application-tool source files.",
    ),
    (
        "multimodal_perception",
        "perception",
        ("src/multimodal_perception.py",),
        "Multimodal perception source file; supported live inputs are not inferred.",
    ),
    (
        "repository_diagnostics",
        "developer_quality",
        ("src/repository_audit.py", "src/self_test.py"),
        "Static repository audit and test-runner source files.",
    ),
    (
        "chat_cli_entrypoint",
        "chat",
        ("src/__main__.py",),
        "Python module entry point for launching chat with python -m src.",
    ),
    (
        "web_chat",
        "chat",
        ("src/web_app.py", "frontend/package.json"),
        "Backend web app and frontend package manifest.",
    ),
    (
        "local_ollama_adapter",
        "inference",
        ("src/ollama_provider.py",),
        "Dedicated local Ollama provider adapter source file.",
    ),
)


def inspect_capabilities(root: str | Path = ".") -> tuple[CapabilityStatus, ...]:
    """Return a deterministic, read-only inventory of known source components."""
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(str(base))

    results = []
    for name, area, required_files, description in CAPABILITY_CATALOG:
        present = tuple(path for path in required_files if (base / path).is_file())
        missing = tuple(path for path in required_files if not (base / path).is_file())
        if not missing:
            status = "source_present"
        elif present:
            status = "partial_source"
        else:
            status = "not_detected"
        results.append(
            CapabilityStatus(
                name=name,
                area=area,
                status=status,
                description=description,
                present_files=present,
                missing_files=missing,
            )
        )
    return tuple(results)


def capabilities_as_json(capabilities: Iterable[CapabilityStatus]) -> str:
    """Serialize capability findings for scripts without implying runtime health."""
    return json.dumps(
        [asdict(item) for item in capabilities],
        indent=2,
        sort_keys=True,
    )
