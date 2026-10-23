import json

import pytest

from src.memory import LongTermMemory


def test_v2_round_trip_preserves_order_and_metadata(tmp_path):
    path = tmp_path / "nested" / "memory.json"
    memory = LongTermMemory()
    memory.remember("first", "one", importance=2.5)
    memory.remember("second", "two", importance=0.5)
    memory.recall("first")
    memory.retrieve("second")

    memory.save(path)
    restored = LongTermMemory(path)

    assert restored.keys() == ["first", "second"]
    assert restored.metadata("first") == {
        "access_count": 1,
        "importance": 2.5,
        "order": 0,
    }
    assert restored.metadata("second")["access_count"] == 1
    assert restored._next_order == 2


def test_legacy_key_value_file_migrates_on_save(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"name": "TARA", "language": "Hindi"}), encoding="utf-8")

    memory = LongTermMemory(path)

    assert memory.keys() == ["name", "language"]
    assert memory.metadata("name") == {"access_count": 0, "importance": 1.0, "order": 0}
    memory.save()
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 2


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(records=[]), "invalid long-term memory format"),
        (lambda p: p["metadata"].update(name={"access_count": -1}), "access_count"),
        (lambda p: p["metadata"].update(name={"importance": float("inf")}), "importance"),
        (lambda p: p.update(next_order=-1), "next_order"),
        (lambda p: p.update(next_order=0), "next_order"),
        (lambda p: p.update(format_version=99), "unsupported"),
    ],
)
def test_invalid_persistence_payload_is_rejected_without_mutating_memory(tmp_path, mutate, message):
    path = tmp_path / "bad.json"
    payload = {
        "format_version": 2,
        "records": {"name": "TARA"},
        "metadata": {"name": {"access_count": 0, "importance": 1.0, "order": 0}},
        "next_order": 1,
    }
    mutate(payload)
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    memory = LongTermMemory()
    memory.remember("existing", "safe")
    before = memory.keys()

    with pytest.raises(ValueError, match=message):
        memory.load(path)

    assert memory.keys() == before
    assert memory.recall("existing") == "safe"


def test_failed_serialization_keeps_previous_file_intact(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text('{"previous": "safe"}', encoding="utf-8")
    memory = LongTermMemory()
    memory.remember("bad", object())

    with pytest.raises(TypeError):
        memory.save(path)

    assert path.read_text(encoding="utf-8") == '{"previous": "safe"}'
    assert list(tmp_path.glob(".memory.json.*.tmp")) == []


def test_unknown_metadata_keys_are_rejected(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(
        json.dumps({
            "format_version": 2,
            "records": {"name": "TARA"},
            "metadata": {"unknown": {"access_count": 0, "importance": 1.0, "order": 0}},
            "next_order": 1,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown key"):
        LongTermMemory(path)
