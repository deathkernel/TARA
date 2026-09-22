from pathlib import Path

from training_preflight import run_preflight


def test_preflight_accepts_repository_algorithm_dataset():
    report = run_preflight(Path("data/algorithm_tasks.jsonl"), context=64)
    assert report["checks"]
    assert any(item["name"] == "dataset_audit" for item in report["checks"])


def test_preflight_rejects_missing_dataset(tmp_path: Path):
    report = run_preflight(tmp_path / "missing.jsonl")
    assert report["ready"] is False
    assert report["checks"][0]["name"] == "dataset_exists"
