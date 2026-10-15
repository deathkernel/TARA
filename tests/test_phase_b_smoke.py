import json
from pathlib import Path

import pytest

from scripts.run_phase_b_smoke import run_phase_b_smoke


def test_phase_b_smoke_trains_evaluates_and_persists_evidence(tmp_path: Path):
    root = tmp_path / "phase-b"
    report = run_phase_b_smoke(root, steps=1, max_new_tokens=2)

    assert report["smoke_passed"] is True
    assert report["dataset_fingerprint"]
    assert report["baseline_training"]["final_step"] > 0
    assert report["candidate_training"]["final_step"] > report["candidate_training"]["start_step"]
    assert report["learning_experiment"]["baseline"]["benchmark"]["case_count"] == 21
    assert report["learning_experiment"]["candidate"] is not None
    assert (root / "baseline.pt").is_file()
    assert (root / "candidate.pt").is_file()
    assert (root / "phase_b_report.json").is_file()


def test_phase_b_smoke_refuses_to_overwrite_existing_artifacts(tmp_path: Path):
    root = tmp_path / "phase-b"
    root.mkdir()
    (root / "baseline.pt").write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_phase_b_smoke(root, steps=1, max_new_tokens=1)

    assert (root / "baseline.pt").read_bytes() == b"keep-me"


def test_phase_b_report_is_machine_readable(tmp_path: Path):
    root = tmp_path / "phase-b"
    report = run_phase_b_smoke(root, steps=1, max_new_tokens=1)
    persisted = json.loads((root / "phase_b_report.json").read_text(encoding="utf-8"))

    assert persisted["schema_version"] == 1
    assert persisted["smoke_passed"] == report["smoke_passed"]
    assert "not a claim" in persisted["interpretation"]
