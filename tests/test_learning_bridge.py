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

class AttributeCandidate:
    language = "python"
    problem = "attribute-based candidate"
    source = "return 42"


def test_bridge_serializes_attribute_and_mapping_candidates(tmp_path):
    with CognitiveMemory(tmp_path / "memory.sqlite") as memory:
        bridge = LearningBridge(memory)
        problem = bridge.record_problem("Serialize candidate data")
        candidates = [
            (AttributeCandidate(), "python", "attribute-based candidate", "return 42"),
            (
                {"language": "rust", "problem": "mapping candidate", "source": "fn solve() {}"},
                "rust",
                "mapping candidate",
                "fn solve() {}",
            ),
        ]

        for candidate, language, expected_problem, expected_source in candidates:
            item_id = bridge.record_benchmark(
                problem_id=problem,
                candidate=candidate,
                benchmark=Benchmark(2, 2, 1.0, 1.0, True),
            )
            stored = memory.get(item_id)
            assert stored["content"]["candidate"]["language"] == language
            assert stored["content"]["candidate"]["problem"] == expected_problem
            assert stored["content"]["candidate"]["source"] == expected_source
            assert stored["attributes"]["language"] == language

