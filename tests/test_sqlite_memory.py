from src.tara_mind.memory.sqlite_store import SQLiteMemoryStore
from src.tara_mind.memory.store import MemoryRecord


def test_sqlite_memory_round_trip(tmp_path):
    path = tmp_path / "memory.db"
    with SQLiteMemoryStore(path) as store:
        store.put(MemoryRecord("name", "TARA", "fact", "test"))
        assert store.get("name").value == "TARA"
        assert store.search("TARA")[0].key == "name"
