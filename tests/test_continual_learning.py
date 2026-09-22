import json

from src.continual_learning import MemoryConsolidator, ReplayCorpusBuilder, VerifiedReplayBuffer
from src.memory import LongTermMemory


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_verified_replay_rejects_unverified_and_deduplicates(tmp_path):
    source = tmp_path / "knowledge.jsonl"
    output = tmp_path / "replay.jsonl"
    write_jsonl(source, [
        {"problem": "sort", "solution": "algorithm A", "verified": True},
        {"problem": "sort", "solution": "algorithm A", "verified": True},
        {"problem": "bad", "solution": "unsafe", "verified": False},
    ])

    report = VerifiedReplayBuffer(seed=7, max_records=10).build(source, output)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert report.input_records == 3
    assert report.eligible_records == 1
    assert report.selected_records == 1
    assert report.duplicates_removed == 2
    assert records[0]["verified"] is True


def test_replay_selection_is_deterministic(tmp_path):
    source = tmp_path / "knowledge.jsonl"
    write_jsonl(source, [
        {"text": f"verified example {i}", "verified": True} for i in range(10)
    ])
    a = VerifiedReplayBuffer(seed=11, max_records=4).load(source)
    b = VerifiedReplayBuffer(seed=11, max_records=4).load(source)
    assert [x.fingerprint for x in VerifiedReplayBuffer(seed=11, max_records=4).select(a)] == [
        x.fingerprint for x in VerifiedReplayBuffer(seed=11, max_records=4).select(b)
    ]


def test_replay_corpus_builder_controls_ratio(tmp_path):
    base = tmp_path / "base.jsonl"
    replay = tmp_path / "replay.jsonl"
    output = tmp_path / "merged.jsonl"
    write_jsonl(base, [{"text": f"base {i}"} for i in range(8)])
    write_jsonl(replay, [{"text": f"knowledge {i}", "verified": True} for i in range(8)])

    report = ReplayCorpusBuilder(replay_ratio=0.25, seed=3).merge(base, replay, output)
    assert report.selected_records == 3
    assert len(output.read_text(encoding="utf-8").splitlines()) == 11


def test_memory_consolidation_retains_frequently_accessed_memories():
    memory = LongTermMemory()
    memory.remember("old", "rare", importance=0.1)
    memory.remember("important", "often used", importance=2.0)
    memory.remember("recent", "used", importance=1.0)
    memory.retrieve("important")
    memory.retrieve("important")
    memory.retrieve("recent")

    report = MemoryConsolidator().consolidate(memory, max_records=2)
    assert report.before == 3
    assert report.after == 2
    assert "important" in report.retained
    assert "old" in report.forgotten


def test_consolidation_can_keep_all_memory():
    memory = LongTermMemory()
    memory.remember("a", "alpha")
    memory.remember("b", "beta")
    report = MemoryConsolidator().consolidate(memory, max_records=5)
    assert report.forgotten == ()
    assert report.after == 2
