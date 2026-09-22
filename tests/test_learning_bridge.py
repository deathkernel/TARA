from dataclasses import dataclass

from src.cognitive_memory import CognitiveMemory
from src.learning_bridge import LearningBridge


@dataclass(frozen=True)
class Candidate:
    problem: str
    language: str
    source: str


@dataclass(frozen=True)
class Benchmark:
    passed: int
    total: int
    correctness: float
    total_runtime_ms: float
    verified: bool
    failures: tuple = ()


def test_bridge_persists_success_and_failure_and_exports(tmp_path):
    with CognitiveMemory(tmp_path / "memory.sqlite") as memory:
        bridge = LearningBridge(memory)
        problem = bridge.record_problem("Find shortest path")
        candidate = Candidate("shortest_path", "rust", "fn solve() {}")

        success = bridge.record_benchmark(
            problem_id=problem,
            candidate=candidate,
            benchmark=Benchmark(3, 3, 1.0, 2.5, True),
        )
        failure = bridge.record_benchmark(
            problem_id=problem,
            candidate=Candidate("shortest_path", "python", "def solve(): pass"),
            benchmark=Benchmark(1, 3, 1 / 3, 1.2, False, ("wrong answer",)),
        )

        assert memory.get(success)["verification"] == "verified"
        assert memory.get(failure)["verification"] == "rejected"
        assert memory.related(problem, relation="verified_solution")[0]["id"] == success
        assert memory.related(problem, relation="failed_attempt")[0]["id"] == failure

        replay = tmp_path / "replay.jsonl"
        failures = tmp_path / "failures.jsonl"
        assert bridge.export_verified_replay(replay) == 1
        assert bridge.export_failures(failures) == 1
        assert replay.read_text(encoding="utf-8").count("\n") == 1
        assert failures.read_text(encoding="utf-8").count("\n") == 1
