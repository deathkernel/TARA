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
