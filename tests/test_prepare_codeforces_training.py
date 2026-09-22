from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from prepare_codeforces_training import prepare


def test_prepare_creates_deterministic_split_and_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "prepared"
    table = pa.table(
        {
            "id": ["2/B", "1/A", "3/C", "4/D"],
            "title": ["B", "A", "C", "D"],
            "description": ["desc B", "desc A", "desc C", "desc D"],
            "input_format": [None] * 4,
            "output_format": [None] * 4,
            "examples": [[], [], [], []],
            "editorial": [None, "editorial A", None, None],
            "rating": [1200, 1000, 1500, 1800],
            "tags": [["math"], ["implementation"], [], ["graphs"]],
            "official_tests": [None, [], [], []],
            "official_tests_complete": [False] * 4,
            "executable": [None] * 4,
        }
    )
    pq.write_table(table, source)

    first = prepare(source, output, validation_ratio=0.25, seed=37)
    train_first = (output / "train.jsonl").read_text(encoding="utf-8")
    validation_first = (output / "validation.jsonl").read_text(encoding="utf-8")

    second_output = tmp_path / "prepared_again"
    second = prepare(source, second_output, validation_ratio=0.25, seed=37)

    assert first == second
    assert train_first == (second_output / "train.jsonl").read_text(encoding="utf-8")
    assert validation_first == (second_output / "validation.jsonl").read_text(encoding="utf-8")
    assert first["source_rows"] == 4
    assert first["usable_rows"] == 4
    assert first["train_rows"] + first["validation_rows"] == 4
    assert first["validation_rows"] == 1
    assert "source_sha256" in first


def test_prepare_drops_rows_without_id_or_description(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "prepared"
    table = pa.table(
        {
            "id": ["1/A", None, "3/C"],
            "title": ["A", "missing id", "missing description"],
            "description": ["valid", "valid", ""],
        }
    )
    pq.write_table(table, source)

    manifest = prepare(source, output, validation_ratio=0.5, seed=1)

    assert manifest["source_rows"] == 3
    assert manifest["usable_rows"] == 1
    assert manifest["train_rows"] + manifest["validation_rows"] == 1
