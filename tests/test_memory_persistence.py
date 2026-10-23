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


def test_malformed_json_does_not_mutate_memory_or_active_path(tmp_path):
    active_path = tmp_path / "active.json"
    corrupt_path = tmp_path / "corrupt.json"
    active_path.write_text("{}", encoding="utf-8")
    corrupt_path.write_text('{"format_version": 2, "records": ', encoding="utf-8")
    memory = LongTermMemory(active_path)
    memory.remember("safe", "value")
    before_keys = memory.keys()
    before_metadata = memory.metadata("safe")

    with pytest.raises(json.JSONDecodeError):
        memory.load(corrupt_path)

    assert memory.keys() == before_keys
    assert memory.metadata("safe") == before_metadata
    assert memory.path == active_path


@pytest.mark.parametrize(
    ("metadata", "next_order", "message"),
    [
        ({"access_count": True, "importance": 1.0, "order": 0}, 1, "access_count"),
        ({"access_count": 0, "importance": float("nan"), "order": 0}, 1, "importance"),
        ({"access_count": 0, "importance": 1.0, "order": True}, 1, "order"),
        ({"access_count": 0, "importance": -1.0, "order": 0}, 1, "importance"),
        ({"access_count": 0, "importance": 1.0, "order": 0}, True, "next_order"),
    ],
)
def test_invalid_metadata_types_are_rejected_before_state_change(
    tmp_path, metadata, next_order, message
):
    path = tmp_path / "invalid.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "records": {"name": "TARA"},
                "metadata": {"name": metadata},
                "next_order": next_order,
            },
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    memory = LongTermMemory()
    memory.remember("keep", "unchanged")
    before = (memory.keys(), memory.metadata("keep"), memory.path)

    with pytest.raises(ValueError, match=message):
        memory.load(path)

    assert (memory.keys(), memory.metadata("keep"), memory.path) == before


def test_versioned_payload_defaults_optional_metadata_safely(tmp_path):
    path = tmp_path / "minimal.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 2,
                "records": {"first": "one", "second": "two"},
            }
        ),
        encoding="utf-8",
    )

    memory = LongTermMemory(path)

    assert memory.keys() == ["first", "second"]
    assert memory.metadata("first") == {
        "access_count": 0,
        "importance": 1.0,
        "order": 0,
    }
    assert memory.metadata("second")["order"] == 1
    assert memory._next_order == 2


def test_successful_save_replaces_snapshot_and_leaves_no_temp_files(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text('{"old": "snapshot"}', encoding="utf-8")
    memory = LongTermMemory()
    memory.remember("new", "snapshot")

    memory.save(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format_version"] == 2
    assert payload["records"] == {"new": "snapshot"}
    assert list(tmp_path.glob(".memory.json.*.tmp")) == []



def test_transactional_memory_mutations_roll_back_all_state_when_save_fails(tmp_path, monkeypatch):
    path = tmp_path / "memory.json"
    path.write_text('{"disk": "snapshot"}', encoding="utf-8")
    memory = LongTermMemory(path)
    memory.remember("first", "one", importance=2.0)
    memory.recall("first")
    before = (memory.keys(), memory.metadata("first"), memory._next_order, memory.path)

    def fail_save():
        raise OSError("disk unavailable")

    monkeypatch.setattr(memory, "save", fail_save)
    with pytest.raises(OSError, match="disk unavailable"):
        memory.remember_and_save("second", "two")
    assert (memory.keys(), memory.metadata("first"), memory._next_order, memory.path) == before
    assert memory.metadata("second") is None

    with pytest.raises(OSError, match="disk unavailable"):
        memory.forget_and_save("first")
    assert (memory.keys(), memory.metadata("first"), memory._next_order, memory.path) == before
    assert path.read_text(encoding="utf-8") == '{"disk": "snapshot"}'


def test_transactional_memory_mutations_persist_successfully(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)

    assert memory.remember_and_save("key", "value") == "value"
    assert LongTermMemory(path).recall("key") == "value"
    assert memory.forget_and_save("key") == "value"
    assert LongTermMemory(path).keys() == []



def test_retrieval_access_counts_persist_and_inform_forgetting(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember("rare", "orchid", importance=1.0)
    memory.remember("useful", "orchid care", importance=1.0)

    assert memory.retrieve("orchid care", min_score=0.75)[0]["key"] == "useful"
    memory.save()

    restored = LongTermMemory(path)
    assert restored.metadata("useful")["access_count"] == 1
    assert restored.metadata("rare")["access_count"] == 0
    assert restored.forget_least_used(1) == ["rare"]
    assert restored.keys() == ["useful"]



def test_transactional_remember_without_destination_rolls_back_everything():
    memory = LongTermMemory()
    memory.remember("existing", "kept", importance=2.0)
    memory.recall("existing")
    before = (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path)

    with pytest.raises(ValueError, match="path is required"):
        memory.remember_and_save("new", "not persisted")

    assert (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path) == before
    assert memory.metadata("new") is None


def test_forget_and_save_missing_key_is_a_noop_without_storage(monkeypatch):
    memory = LongTermMemory()
    memory.remember("existing", "kept")
    before = (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path)
    save_calls = []

    def track_save(*args, **kwargs):
        save_calls.append((args, kwargs))

    monkeypatch.setattr(memory, "save", track_save)

    assert memory.forget_and_save("missing") is None
    assert save_calls == []
    assert (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path) == before


@pytest.mark.parametrize("invalid_key", ["", None, 42])
def test_transactional_remember_rejects_invalid_keys_without_partial_mutation(invalid_key):
    memory = LongTermMemory()
    memory.remember("existing", "kept")
    before = (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path)

    with pytest.raises(ValueError, match="key"):
        memory.remember_and_save(invalid_key, "value")

    assert (memory.keys(), memory.metadata("existing"), memory._next_order, memory.path) == before


def test_transactional_update_preserves_usage_history_after_reload(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember_and_save("topic", "old", importance=1.0)
    memory.recall("topic")

    memory.remember_and_save("topic", "updated", importance=3.0)

    restored = LongTermMemory(path)
    assert restored.recall("topic") == "updated"
    assert restored.metadata("topic") == {
        "access_count": 2,
        "importance": 3.0,
        "order": 0,
    }
