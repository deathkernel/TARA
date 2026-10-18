"""Read-only capability inventory for the current TARA checkout."""
from __future__ import annotations
import ast
import importlib.util
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

@dataclass(frozen=True)
class CapabilityStatus:
    name: str
    category: str
    source_present: bool
    dependency: str | None
    dependency_installed: bool | None
    configured: bool | None
    enabled: bool
    available: bool
    runtime_wired: bool
    details: str
    def to_dict(self) -> dict[str, object]:
        return asdict(self)

def _module_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False

def _references(path: Path, symbol: str) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False
    return any(
        (isinstance(node, ast.Name) and node.id == symbol) or
        (isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            (item.asname or item.name.split(".")[0]) == symbol for item in node.names
        ))
        for node in ast.walk(tree)
    )

def collect_capabilities(root=".", environ: Mapping[str, str] | None = None) -> list[CapabilityStatus]:
    """Report source, dependency, configuration, and default-runtime wiring."""
    base = Path(root).resolve()
    env = os.environ if environ is None else environ
    entry = base / "src" / "__main__.py"
    specs = (
        ("chat_runtime", "runtime", ("src/__main__.py",), None, (), {}, None),
        ("cognitive_core", "component_group", ("src/brain.py","src/tara_core.py","src/unified_cognitive_loop.py"), None, (), {}, "TaraCore"),
        ("ollama_provider", "llm_provider_adapter", ("src/ollama_provider.py","src/llm_provider_factory.py"), None, ("TARA_LLM_BACKEND",), {"TARA_LLM_BACKEND": ("ollama","llama")}, "provider_from_env"),
        ("groq_provider", "llm_provider_adapter", ("src/groq_provider.py",), "groq", ("GROQ_API_KEY",), {}, "GroqProvider"),
        ("elevenlabs_tts", "voice_output_adapter", ("src/elevenlabs_tts.py",), "elevenlabs", ("ELEVENLABS_API_KEY",), {}, "ElevenLabsTTS"),
        ("public_api_tools", "read_only_tools", ("src/public_api_tools.py",), None, (), {}, "PublicAPITools"),
    )
    result = []
    for name, category, paths, dep, keys, allowed, symbol in specs:
        source = all((base / p).is_file() for p in paths)
        installed = None if dep is None else _module_installed(dep)
        configured = None if not keys else (
            all(env.get(k, "").strip().lower() in values for k, values in allowed.items())
            if allowed else all(bool(env.get(k, "").strip()) for k in keys)
        )
        wired = bool(entry.is_file() and symbol and _references(entry, symbol))
        if name == "chat_runtime":
            wired = source
            enabled = available = source
            detail = "entry point present" if source else "no src/__main__.py entry point found"
        elif category == "component_group":
            enabled, available = wired and source, source
            detail = "source modules present; default-runtime wiring is not confirmed" if source and not wired else (
                "source modules wired into the default runtime" if wired else "one or more required source modules are missing"
            )
        else:
            ready = source and installed is not False and configured is not False
            enabled, available = wired and ready, ready
            missing = [k for k in keys if not env.get(k, "").strip()]
            detail = ("source files missing from this checkout" if not source else
                f"optional dependency '{dep}' is not installed" if installed is False else
                f"{keys[0]} must be set to one of: " + ", ".join(allowed[keys[0]]) if configured is False and allowed else
                "missing required configuration: " + ", ".join(missing) if configured is False else
                "adapter/tool prerequisites are met, but default-runtime wiring was not detected" if not wired else
                "prerequisites and default-runtime wiring are present")
        result.append(CapabilityStatus(name, category, source, dep, installed, configured, enabled, available, wired, detail))
    return result

def capability_report(root=".", environ: Mapping[str, str] | None = None) -> dict[str, object]:
    """Return status flags without environment-variable values."""
    base = Path(root).resolve()
    env = os.environ if environ is None else environ
    items = collect_capabilities(base, env)
    entry = base / "src" / "__main__.py"
    selector = env.get("TARA_LLM_BACKEND", "").strip().lower()
    providers = {"groq": ("GroqProvider","groq"), "ollama": ("provider_from_env","ollama"), "llama": ("provider_from_env","ollama")}
    active = None
    if entry.is_file() and selector in providers and _references(entry, providers[selector][0]):
        active = providers[selector][1]
    return {"root": str(base), "runtime_available": entry.is_file(), "active_provider": active,
            "active_components": [item.name for item in items if item.enabled],
            "capabilities": [item.to_dict() for item in items]}
