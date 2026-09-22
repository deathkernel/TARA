import json

from src.training_dataset_mixer import TargetedDatasetMixer
from src.training_pipeline import load_training_texts


def test_targeted_records_preserve_solution_and_tests(tmp_path):
    path = tmp_path / "targeted.jsonl"
    path.write_text(json.dumps({"prompt": "Compute 2 + 2", "solution": 4, "tests": "expected=4"}) + "\n", encoding="utf-8")
    texts = load_training_texts(path)
    assert "Prompt: Compute 2 + 2" in texts[0]
    assert "Answer: 4" in texts[0]
    assert "Tests: expected=4" in texts[0]


def test_mixer_is_deterministic_and_does_not_modify_sources(tmp_path):
    base = tmp_path / "base.jsonl"
    targeted = tmp_path / "targeted.jsonl"
    output = tmp_path / "mixed.jsonl"
    base.write_text(json.dumps({"text": "base-1"}) + "\n" + json.dumps({"text": "base-2"}) + "\n", encoding="utf-8")
    targeted.write_text(json.dumps({"prompt": "target-1", "solution": "answer", "tests": "ok"}) + "\n", encoding="utf-8")
    before_base, before_targeted = base.read_text(), targeted.read_text()
    report = TargetedDatasetMixer(targeted_ratio=0.5, seed=7).mix(base, targeted, output)
    assert report.base_records == 2
    assert report.targeted_records == 1
    assert report.selected_targeted == 1
    assert output.exists()
    assert base.read_text() == before_base
    assert targeted.read_text() == before_targeted
    assert len(load_training_texts(output)) == 3


def test_targeted_ratio_zero_keeps_only_base(tmp_path):
    base = tmp_path / "base.jsonl"
    targeted = tmp_path / "targeted.jsonl"
    output = tmp_path / "mixed.jsonl"
    base.write_text(json.dumps({"text": "base"}) + "\n", encoding="utf-8")
    targeted.write_text(json.dumps({"prompt": "target", "solution": "answer"}) + "\n", encoding="utf-8")
    report = TargetedDatasetMixer(targeted_ratio=0.0).mix(base, targeted, output)
    assert report.selected_targeted == 0
    assert load_training_texts(output) == ["base"]
