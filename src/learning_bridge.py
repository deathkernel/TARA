"""Bridge algorithm research outcomes into TARA's persistent cognitive memory.

The bridge records both successful and failed attempts. Verified solutions can
also be exported as training/replay examples, while failures stay available
for targeted learning instead of silently disappearing.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .cognitive_memory import CognitiveMemory
from .polyglot.knowledge import KnowledgeRecord


class LearningBridge:
    """Persist discovery experiences and export verified replay examples."""

    def __init__(self, memory: CognitiveMemory) -> None:
        self.memory = memory

    def record_problem(self, problem: str, *, metadata: dict[str, Any] | None = None) -> str:
        return self.memory.remember(
            kind="problem",
            domain="algorithm_research",
            content={"description": problem},
            provenance={"source": "learning_bridge"},
            attributes=metadata or {},
        )

    def record_benchmark(
        self,
        *,
        problem_id: str,
        candidate: Any,
        benchmark: Any,
    ) -> str:
        verified = bool(benchmark.verified)
        kind = "verified_algorithm" if verified else "algorithm_failure"
        content = {
            "problem_id": problem_id,
            "candidate": asdict(candidate) if hasattr(candidate, "__dataclass_fields__") else candidate,
            "benchmark": {
                "passed": benchmark.passed,
                "total": benchmark.total,
                "correctness": benchmark.correctness,
                "runtime_ms": benchmark.total_runtime_ms,
                "verified": verified,
                "failures": list(benchmark.failures),
            },
        }
        item_id = self.memory.remember(
            kind=kind,
            domain="algorithm_research",
            content=content,
            confidence=float(benchmark.correctness),
            verification="verified" if verified else "rejected",
            provenance={"source": "polyglot_benchmark"},
            attributes={"language": getattr(candidate, "language", "unknown")},
        )
        self.memory.link(problem_id, "verified_solution" if verified else "failed_attempt", item_id)
        return item_id

    def record_knowledge(self, record: KnowledgeRecord, *, problem_id: str | None = None) -> str:
        item_id = self.memory.remember(
            kind="verified_knowledge",
            domain="algorithm_research",
            content=asdict(record),
            confidence=float(record.correctness),
            verification="verified" if record.verified else "rejected",
            provenance={"source": record.provenance, "fingerprint": record.fingerprint},
            attributes={"language": record.language},
            knowledge_id=record.fingerprint,
        )
        if problem_id is not None:
            self.memory.link(problem_id, "verified_knowledge", item_id)
        return item_id

    @staticmethod
    def _replay_payload(item: dict[str, Any]) -> dict[str, Any]:
        """Normalize memory records into the trainer's problem/solution shape."""
        content = item["content"]
        if item["kind"] == "verified_knowledge":
            return content
        candidate = content["candidate"]
        benchmark = content["benchmark"]
        return {
            "problem": candidate.get("problem", content.get("problem_id", "")),
            "solution": candidate.get("source", ""),
            "language": candidate.get("language", "unknown"),
            "tests": benchmark.get("failures", []),
            "correctness": benchmark.get("correctness", 0.0),
            "runtime_ms": benchmark.get("runtime_ms", 0.0),
            "verification": benchmark.get("verified", False),
        }

    def export_verified_replay(self, path: str | Path, *, limit: int = 10000) -> int:
        """Export verified memory as deterministic JSONL training/replay data."""
        knowledge = self.memory.search(kind="verified_knowledge", verification="verified", limit=limit)
        algorithms = self.memory.search(kind="verified_algorithm", verification="verified", limit=limit)
        by_id = {item["id"]: item for item in (*knowledge, *algorithms)}
        records = sorted(by_id.values(), key=lambda item: item["id"])[:limit]
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(self._replay_payload(item), ensure_ascii=False, sort_keys=True) + "\n")
        return len(records)

    def export_failures(self, path: str | Path, *, limit: int = 10000) -> int:
        """Export rejected attempts for targeted failure-focused learning."""
        records = self.memory.search(kind="algorithm_failure", verification="rejected", limit=limit)
        records.sort(key=lambda item: item["id"])
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item["content"], ensure_ascii=False, sort_keys=True) + "\n")
        return len(records)
