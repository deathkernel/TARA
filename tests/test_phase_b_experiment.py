from pathlib import Path

import pytest

from scripts.run_phase_b_experiment import run_phase_b_experiment


def test_phase_b_experiment_rejects_missing_baseline_and_dataset(tmp_path: Path):
    baseline = tmp_path / "missing.pt"
    dataset = tmp_path / "reviewed.jsonl"
    with pytest.raises(FileNotFoundError, match="baseline checkpoint"):
        run_phase_b_experiment(baseline, dataset, tmp_path / "candidate.pt")


def test_phase_b_experiment_never_allows_baseline_overwrite(tmp_path: Path):
    baseline = tmp_path / "baseline.pt"
    baseline.write_bytes(b"protected-baseline")
    dataset = tmp_path / "reviewed.jsonl"
    dataset.write_text('{"text":"A sufficiently long reviewed training example."}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="must differ"):
        run_phase_b_experiment(baseline, dataset, baseline)

    assert baseline.read_bytes() == b"protected-baseline"


def test_phase_b_experiment_refuses_existing_candidate(tmp_path: Path):
    baseline = tmp_path / "baseline.pt"
    dataset = tmp_path / "reviewed.jsonl"
    candidate = tmp_path / "candidate.pt"
    baseline.write_bytes(b"baseline")
    dataset.write_text('{"text":"A sufficiently long reviewed training example."}\n', encoding="utf-8")
    candidate.write_bytes(b"candidate-protected")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_phase_b_experiment(baseline, dataset, candidate)

    assert baseline.read_bytes() == b"baseline"
    assert candidate.read_bytes() == b"candidate-protected"
