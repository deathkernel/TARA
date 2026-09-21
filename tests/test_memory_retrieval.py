from src.memory import LongTermMemory, lexical_relevance


def test_lexical_relevance_is_normalized_and_case_insensitive():
    assert lexical_relevance("Python security", "python tools for security") == 1.0
    assert lexical_relevance("missing", "nothing here") == 0.0
    assert lexical_relevance("", "anything") == 0.0


def test_retrieve_ranks_relevant_memories_deterministically():
    memory = LongTermMemory()
    memory.remember("python security", "Study Python security fundamentals")
    memory.remember("python cooking", "Learn Python recipes")
    memory.remember("music", "Listen to music")

    results = memory.retrieve("python security")
    assert [item["key"] for item in results] == ["python security", "python cooking"]
    assert results[0]["score"] == 1.0
    assert results[1]["score"] == 0.5


def test_retrieve_supports_limit_and_min_score():
    memory = LongTermMemory()
    memory.remember("alpha beta", "first")
    memory.remember("alpha", "second")
    memory.remember("gamma", "third")

    results = memory.retrieve("alpha beta", limit=1, min_score=0.5)
    assert len(results) == 1
    assert results[0]["key"] == "alpha beta"


def test_retrieve_increments_access_count():
    memory = LongTermMemory()
    memory.remember("python", "learn python")
    assert memory.metadata("python")["access_count"] == 0
    memory.retrieve("python")
    assert memory.metadata("python")["access_count"] == 1
    memory.recall("python")
    assert memory.metadata("python")["access_count"] == 2


def test_remember_updates_value_and_importance_without_resetting_history():
    memory = LongTermMemory()
    memory.remember("fact", "old", importance=0.5)
    memory.recall("fact")
    memory.remember("fact", "new", importance=2.0)

    assert memory.recall("fact") == "new"
    metadata = memory.metadata("fact")
    assert metadata["access_count"] == 2
    assert metadata["importance"] == 2.0


def test_forget_least_used_prefers_low_usage_then_low_importance():
    memory = LongTermMemory()
    memory.remember("used", "keep me", importance=1.0)
    memory.remember("important", "keep me", importance=10.0)
    memory.remember("unused", "remove me", importance=0.1)
    memory.recall("used")

    forgotten = memory.forget_least_used(2)
    assert forgotten == ["unused"]
    assert set(memory.keys()) == {"used", "important"}


def test_forget_least_used_handles_zero_capacity():
    memory = LongTermMemory()
    memory.remember("a", 1)
    memory.remember("b", 2)
    assert memory.forget_least_used(0) == ["a", "b"]
    assert memory.keys() == []


def test_v2_memory_round_trip_preserves_metadata(tmp_path):
    path = tmp_path / "memory.json"
    memory = LongTermMemory(path)
    memory.remember("name", "TARA", importance=3.0)
    memory.retrieve("name")
    memory.save()

    restored = LongTermMemory(path)
    assert restored.recall("name") == "TARA"
    assert restored.metadata("name")["access_count"] == 2
    assert restored.metadata("name")["importance"] == 3.0


def test_loads_original_plain_key_value_format(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text('{"name": "TARA"}', encoding="utf-8")

    memory = LongTermMemory(path)
    assert memory.recall("name") == "TARA"
    assert memory.metadata("name")["access_count"] == 1
