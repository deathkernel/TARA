from pathlib import Path

import pytest

from src.dataset_pipeline import DatasetBuilder, DatasetError, DatasetSource


def test_jsonl_ingestion_and_deduplication(tmp_path: Path) -> None:
    source = tmp_path / "examples.jsonl"
    source.write_text(
        '{"id": 1, "text": "hello world"}\n'
        '{"id": 2, "content": "hello world"}\n'
        '{"id": 3, "prompt": "sort this list"}\n',
        encoding="utf-8",
    )

    records, manifest = DatasetBuilder().prepare([source])

    assert [record.text for record in records] == ["hello world", "sort this list"]
    assert manifest.records == 3
    assert manifest.unique_records == 2
    assert manifest.duplicates_removed == 1


def test_json_object_with_data_field(tmp_path: Path) -> None:
    source = tmp_path / "data.json"
    source.write_text('{"data": [{"text": "one"}, {"text": "two"}]}', encoding="utf-8")

    records = DatasetSource.read(source)
    assert [record.text for record in records] == ["one", "two"]


def test_text_source(tmp_path: Path) -> None:
    source = tmp_path / "data.txt"
    source.write_text("one\n\ntwo\n", encoding="utf-8")

    records = DatasetSource.read(source)
    assert [record.text for record in records] == ["one", "two"]


def test_invalid_record_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text('{"data": [{"id": 1}]}', encoding="utf-8")

    with pytest.raises(DatasetError):
        DatasetSource.read(source)


def test_split_preserves_all_records() -> None:
    records = [
        # Minimal records are enough to exercise deterministic slicing.
        type("R", (), {"text": str(i)})() for i in range(10)
    ]
    train, validation, test = DatasetBuilder.split(records, 0.2, 0.2)
    assert len(train) + len(validation) + len(test) == 10
    assert len(validation) == 2
    assert len(test) == 2
