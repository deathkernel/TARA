from src.memory import LongTermMemory, WorkingMemory


def test_working_memory_is_bounded_and_keeps_recent_items():
    memory = WorkingMemory(capacity=3)
    memory.add("a")
    memory.add("b")
    memory.add("c")
    memory.add("d")
    assert memory.recent() == ["b", "c", "d"]
    assert memory.recent(2) == ["c", "d"]


def test_working_memory_clear():
    memory = WorkingMemory(capacity=2)
    memory.add("a")
    memory.clear()
    assert len(memory) == 0
    assert memory.recent() == []


def test_long_term_memory_round_trip(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember("name", "TARA")
    memory.remember("version", 1)
    memory.save()

    restored = LongTermMemory(path)
    assert restored.recall("name") == "TARA"
    assert restored.recall("version") == 1
    assert restored.recall("missing", "fallback") == "fallback"


def test_long_term_memory_forget():
    memory = LongTermMemory()
    memory.remember("fact", "value")
    assert memory.forget("fact") == "value"
    assert memory.recall("fact") is None
    assert memory.forget("fact") is None


def test_long_term_memory_rejects_non_object_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    try:
        LongTermMemory(path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("invalid memory payload was accepted")



def test_unicode_memory_retrieval_supports_hindi_and_accented_text():
    memory = LongTermMemory()
    memory.remember("भाषा", "मुझे हिंदी में जवाब देना")
    memory.remember("cafe", "préférer un café crème")

    hindi_results = memory.retrieve("हिंदी जवाब", limit=5)
    french_results = memory.retrieve("café crème", limit=5)

    assert hindi_results
    assert hindi_results[0]["key"] == "भाषा"
    assert french_results
    assert french_results[0]["key"] == "cafe"


def test_retrieval_limit_requires_non_negative_integer():
    import pytest

    memory = LongTermMemory()
    memory.remember("language", "Hindi")
    assert memory.retrieve("Hindi", limit=0) == []
    for invalid in (-1, True, 1.5, "2"):
        with pytest.raises(ValueError, match="limit"):
            memory.retrieve("Hindi", limit=invalid)


def test_retrieval_min_score_requires_finite_number_in_range():
    import pytest

    memory = LongTermMemory()
    memory.remember("language", "Hindi")
    assert memory.retrieve("Hindi", min_score=1.0)
    for invalid in (-0.1, 1.1, True, float("nan"), float("inf"), "0.5"):
        with pytest.raises(ValueError, match="min_score"):
            memory.retrieve("Hindi", min_score=invalid)


def test_equal_relevance_ranking_is_stable_and_limit_is_respected():
    memory = LongTermMemory()
    memory.remember("first", "Hindi")
    memory.remember("second", "Hindi")
    memory.remember("third", "English")

    first_run = memory.retrieve("Hindi language", limit=2)
    second_run = memory.retrieve("Hindi language", limit=2)

    assert [item["key"] for item in first_run] == ["first", "second"]
    assert [item["key"] for item in second_run] == ["first", "second"]
    assert len(memory.retrieve("Hindi", limit=1)) == 1



def test_atomic_save_preserves_previous_file_when_value_is_not_json_serializable(tmp_path):
    import json
    import pytest

    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember("stable", "last good value")
    memory.save()
    original = path.read_text(encoding="utf-8")

    memory.remember("unsupported", object())
    with pytest.raises(TypeError):
        memory.save()

    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(".memory.json.*.tmp"))
    assert json.loads(path.read_text(encoding="utf-8"))["records"] == {
        "stable": "last good value"
    }


def test_invalid_metadata_does_not_partially_replace_live_memory(tmp_path):
    import json
    import pytest

    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({
        "format_version": 2,
        "records": {"new": "value"},
        "metadata": {"new": {"access_count": "many", "importance": 1.0, "order": 0}},
        "next_order": 1,
    }), encoding="utf-8")
    memory = LongTermMemory()
    memory.remember("existing", "keep me")

    with pytest.raises(ValueError, match="invalid metadata"):
        memory.load(path)

    assert memory.keys() == ["existing"]
    assert memory.recall("existing") == "keep me"
    assert memory.path is None


def test_unknown_memory_format_version_is_rejected_without_state_mutation(tmp_path):
    import json
    import pytest

    path = tmp_path / "future.json"
    path.write_text(json.dumps({
        "format_version": 99, "records": {}, "metadata": {}, "next_order": 0
    }), encoding="utf-8")
    memory = LongTermMemory()
    memory.remember("existing", "keep me")

    with pytest.raises(ValueError, match="unsupported.*version"):
        memory.load(path)

    assert memory.keys() == ["existing"]


def test_legacy_key_value_memory_format_still_loads(tmp_path):
    import json

    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"language": "Hinglish", "editor": "DaVinci"}), encoding="utf-8")

    memory = LongTermMemory(path)

    assert memory.recall("language") == "Hinglish"
    assert memory.recall("editor") == "DaVinci"
    assert memory.metadata("language") == {
        "access_count": 1, "importance": 1.0, "order": 0
    }
