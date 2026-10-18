import json
from pathlib import Path

from src import capabilities
from src.doctor import main


def _write(root: Path, relative: str, content: str = "# module\n"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_registry_distinguishes_adapter_readiness_from_runtime_wiring(tmp_path, monkeypatch):
    for path in ("src/groq_provider.py", "src/elevenlabs_tts.py", "src/brain.py", "src/tara_core.py", "src/unified_cognitive_loop.py"):
        _write(tmp_path, path)
    monkeypatch.setattr(capabilities, "_module_installed", lambda _: True)
    report = capabilities.capability_report(tmp_path, {
        "GROQ_API_KEY": "alpha-marker",
        "ELEVENLABS_API_KEY": "beta-marker",
        "ELEVENLABS_VOICE_ID": "voice-id",
    })
    by_name = {item["name"]: item for item in report["capabilities"]}
    assert by_name["groq_provider"]["configured"] is True
    assert by_name["groq_provider"]["available"] is True
    assert by_name["groq_provider"]["runtime_wired"] is False
    assert by_name["groq_provider"]["enabled"] is False
    assert by_name["elevenlabs_tts"]["available"] is True
    assert by_name["chat_runtime"]["source_present"] is False
    assert by_name["ollama_provider"]["source_present"] is False
    assert by_name["public_api_tools"]["source_present"] is False
    serialized = json.dumps(report)
    assert "alpha-marker" not in serialized
    assert "beta-marker" not in serialized


def test_registry_reports_missing_optional_configuration(tmp_path, monkeypatch):
    _write(tmp_path, "src/groq_provider.py")
    monkeypatch.setattr(capabilities, "_module_installed", lambda _: True)
    report = capabilities.collect_capabilities(tmp_path, {})
    groq = next(item for item in report if item.name == "groq_provider")
    assert groq.source_present is True
    assert groq.dependency_installed is True
    assert groq.configured is False
    assert groq.available is False
    assert "GROQ_API_KEY" in groq.details


def test_doctor_status_command_emits_json_without_environment_values(tmp_path, monkeypatch, capsys):
    _write(tmp_path, "src/groq_provider.py")
    monkeypatch.setattr(capabilities, "_module_installed", lambda _: True)
    monkeypatch.setenv("GROQ_API_KEY", "gamma-marker")
    assert main(["status", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["capabilities"]
    assert "gamma-marker" not in output


def test_registry_only_reports_provider_active_when_selector_and_wiring_match(tmp_path, monkeypatch):
    _write(tmp_path, "src/__main__.py", "from .groq_provider import GroqProvider\n")
    _write(tmp_path, "src/groq_provider.py")
    monkeypatch.setattr(capabilities, "_module_installed", lambda _: True)
    report = capabilities.capability_report(tmp_path, {"GROQ_API_KEY": "delta-marker", "TARA_LLM_BACKEND": "groq"})
    groq = next(item for item in report["capabilities"] if item["name"] == "groq_provider")
    assert report["active_provider"] == "groq"
    assert groq["runtime_wired"] is True
    assert groq["enabled"] is True
