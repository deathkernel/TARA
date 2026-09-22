from pathlib import Path

from src.cognitive_memory import CognitiveMemory
from src.continual_learning_cycle import ContinualLearningCycle


class Candidate:
    language = "python"
    problem = "sum two numbers"
    source = "return a + b"


class Benchmark:
    verified = True
    passed = 2
    total = 2
    correctness = 1.0
    total_runtime_ms = 1.25
    failures = []


class FailedBenchmark:
    verified = False
    passed = 1
    total = 2
    correctness = 0.5
    total_runtime_ms = 2.5
    failures = ["case 2 failed"]


def test_prepare_exports_verified_and_failure_memory(tmp_path: Path) -> None:
    memory = CognitiveMemory(tmp_path / "tara.db")
    bridge = __import__("src.learning_bridge", fromlist=["LearningBridge"]).LearningBridge(memory)
    problem_id = bridge.record_problem("sum two numbers")
    bridge.record_benchmark(problem_id=problem_id, candidate=Candidate(), benchmark=Benchmark())
    bridge.record_benchmark(problem_id=problem_id, candidate=Candidate(), benchmark=FailedBenchmark())

    base = tmp_path / "base.jsonl"
    base.write_text('{"text":"base training example"}\n', encoding="utf-8")
    cycle = ContinualLearningCycle(memory, replay_ratio=1.0, failure_ratio=1.0)
    report = cycle.prepare(
        base,
        tmp_path / "mixed.jsonl",
        replay_path=tmp_path / "replay.jsonl",
        failures_path=tmp_path / "failures.jsonl",
    )

    assert report.base_records == 1
    assert report.replay_records == 1
    assert report.failure_records == 1
    assert report.selected_replay == 1
    assert report.selected_failures == 1
    assert report.dataset_fingerprint
    mixed = (tmp_path / "mixed.jsonl").read_text(encoding="utf-8")
    assert "sum two numbers" in mixed
    assert "Failed approach" in mixed


def test_prepare_is_deterministic(tmp_path: Path) -> None:
    memory = CognitiveMemory(tmp_path / "tara.db")
    bridge = __import__("src.learning_bridge", fromlist=["LearningBridge"]).LearningBridge(memory)
    problem_id = bridge.record_problem("deterministic problem")
    bridge.record_benchmark(problem_id=problem_id, candidate=Candidate(), benchmark=Benchmark())
    base = tmp_path / "base.jsonl"
    base.write_text('{"text":"base"}\n', encoding="utf-8")

    cycle = ContinualLearningCycle(memory, seed=37, replay_ratio=1.0)
    first = cycle.prepare(base, tmp_path / "one.jsonl", replay_path=tmp_path / "one.replay.jsonl", failures_path=tmp_path / "one.fail.jsonl")
    second = cycle.prepare(base, tmp_path / "two.jsonl", replay_path=tmp_path / "two.replay.jsonl", failures_path=tmp_path / "two.fail.jsonl")

    assert first.dataset_fingerprint == second.dataset_fingerprint
    assert (tmp_path / "one.jsonl").read_text(encoding="utf-8") == (tmp_path / "two.jsonl").read_text(encoding="utf-8")
