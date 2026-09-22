import json
from pathlib import Path

from src.targeted_training import TargetedTrainingBuilder, build_targeted_dataset


def test_builder_creates_deterministic_variants():
    diagnosis = {
        "failures": [{
            "case_id": "coding-arithmetic",
            "category": "coding",
            "prompt": "Compute 17 * 3",
            "expected": 51,
            "kind": "incorrect_answer",
        }]
    }
    first = TargetedTrainingBuilder(variants_per_failure=3).build(diagnosis)
    second = TargetedTrainingBuilder(variants_per_failure=3).build(diagnosis)
    assert [item.as_dict() for item in first] == [item.as_dict() for item in second]
    assert len(first) == 3
    assert all(item.provenance == "synthetic-from-benchmark-failure" for item in first)


def test_build_targeted_dataset_writes_jsonl(tmp_path: Path):
    diagnosis_path = tmp_path / "diagnosis.json"
    output_path = tmp_path / "targeted.jsonl"
    diagnosis_path.write_text(json.dumps({
        "failures": [{
            "case_id": "memory-token",
            "category": "memory",
            "prompt": "Recall token TARA-37",
            "expected": "TARA-37",
            "kind": "incorrect_answer",
        }]
    }), encoding="utf-8")
    examples = build_targeted_dataset(diagnosis_path, output_path, variants_per_failure=2)
    assert len(examples) == 2
    assert len(output_path.read_text(encoding="utf-8").splitlines()) == 2
