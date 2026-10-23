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
