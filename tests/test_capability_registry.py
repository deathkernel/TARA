import json

import pytest

from src.capability_registry import capabilities_as_json, inspect_capabilities


def test_registry_reports_source_presence_without_importing_modules(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "brain.py").write_text("# source only\n", encoding="utf-8")
    (tmp_path / "src" / "structured_reasoning.py").write_text("# source only\n", encoding="utf-8")

    results = {item.name: item for item in inspect_capabilities(tmp_path)}

    assert results["core_reasoning"].status == "source_present"
    assert results["core_reasoning"].present_files == (
        "src/brain.py",
        "src/structured_reasoning.py",
    )
    assert results["core_reasoning"].missing_files == ()


def test_registry_distinguishes_partial_and_missing_components(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "advanced_planning.py").write_text("# present\n", encoding="utf-8")

    results = {item.name: item for item in inspect_capabilities(tmp_path)}

    assert results["planning"].status == "partial_source"
    assert results["planning"].missing_files == ("src/goal_progress.py",)
    assert results["web_chat"].status == "not_detected"


def test_registry_output_is_deterministic_and_machine_readable(tmp_path):
    first = capabilities_as_json(inspect_capabilities(tmp_path))
    second = capabilities_as_json(inspect_capabilities(tmp_path))

    assert first == second
    decoded = json.loads(first)
    assert decoded[0]["name"] == "core_reasoning"
    assert decoded[0]["status"] == "not_detected"
    assert decoded[0]["missing_files"]


def test_registry_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        inspect_capabilities(tmp_path / "missing")
