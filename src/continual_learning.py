"""Controlled continual-learning and memory-consolidation primitives for TARA.

Phase 15 keeps learning explicit and evidence-gated. Verified knowledge can be
mixed into a new training corpus through a bounded replay buffer, while long-
term memory can be consolidated deterministically using access and importance
signals. This module never executes generated code and never starts training by
itself.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReplayRecord:
    """A replay example with provenance and verification state."""

    text: str
    fingerprint: str
    source: str
    verified: bool = True
    provenance: str = "verified-knowledge"


@dataclass(frozen=True)
class ReplayReport:
    """Summary of a replay-corpus construction operation."""

    input_records: int
    eligible_records: int
    selected_records: int
    duplicates_removed: int
    output_path: str
    fingerprint: str


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _knowledge_text(item: dict) -> str:
    if "problem" in item and "solution" in item:
        return (
            "Problem: " + str(item["problem"])
            + "\nApproach: " + str(item["solution"])
            + "\nTests: " + str(item.get("tests", ""))
        )
    value = item.get("text", item.get("content", item.get("prompt")))
    if value is None:
        raise ValueError("knowledge record has no text/content/prompt or problem/solution")
    return str(value).strip()


class VerifiedReplayBuffer:
    """Build a bounded, deterministic replay set from verified knowledge.

    The source JSONL is treated as data only. Records marked ``verified=false``
    are rejected, and duplicate text is removed before sampling. Sampling uses
    a seeded shuffle so experiments are reproducible.
    """

    def __init__(self, *, seed: int = 42, max_records: int = 1024) -> None:
        if max_records <= 0:
            raise ValueError("max_records must be positive")
        self.seed = int(seed)
        self.max_records = int(max_records)

    def load(self, path: str | Path) -> tuple[ReplayRecord, ...]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        records: list[ReplayRecord] = []
        seen: set[str] = set()
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"knowledge record at line {line_number} must be an object")
            verified = item.get("verified", True)
            if verified is not True:
                continue
            text = _knowledge_text(item)
            if not text:
                continue
            fingerprint = _fingerprint(text)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            records.append(
                ReplayRecord(
                    text=text,
                    fingerprint=fingerprint,
                    source=str(item.get("source", item.get("provenance", source))),
                )
            )
        return tuple(records)

    def select(self, records: Iterable[ReplayRecord]) -> tuple[ReplayRecord, ...]:
        items = list(records)
        rng = random.Random(self.seed)
        rng.shuffle(items)
        return tuple(items[: self.max_records])

    def write(self, records: Iterable[ReplayRecord], path: str | Path) -> ReplayReport:
        items = tuple(records)
        selected = self.select(items)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for record in selected:
                handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
        fingerprint = hashlib.sha256(
            "\n".join(record.fingerprint for record in selected).encode("utf-8")
        ).hexdigest()
        return ReplayReport(
            input_records=len(items),
            eligible_records=len(items),
            selected_records=len(selected),
            duplicates_removed=0,
            output_path=str(destination),
            fingerprint=fingerprint,
        )

    def build(self, path: str | Path, output: str | Path) -> ReplayReport:
        source = Path(path)
        raw_lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        records = self.load(source)
        return self.write(records, output)._replace(  # type: ignore[attr-defined]
            input_records=len(raw_lines)
        )


class ReplayCorpusBuilder:
    """Merge a base corpus with a bounded verified replay corpus."""

    def __init__(self, *, replay_ratio: float = 0.25, seed: int = 42) -> None:
        if not 0.0 <= replay_ratio <= 1.0:
            raise ValueError("replay_ratio must be in [0, 1]")
        self.replay_ratio = float(replay_ratio)
        self.seed = int(seed)

    @staticmethod
    def _read(path: str | Path) -> list[str]:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        records = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = _knowledge_text(item)
            else:
                raise ValueError("corpus record must be a string or object")
            if text:
                records.append(text)
        return records

    def merge(self, base_path: str | Path, replay_path: str | Path, output: str | Path) -> ReplayReport:
        base = self._read(base_path)
        replay = self._read(replay_path)
        if not base and not replay:
            raise ValueError("both base and replay corpora are empty")
        if not replay or self.replay_ratio == 0.0:
            selected = []
        else:
            count = max(1, round(len(base) * self.replay_ratio / max(1e-9, 1.0 - self.replay_ratio))) if base else len(replay)
            selected = list(replay)
            random.Random(self.seed).shuffle(selected)
            selected = selected[:count]
        merged = base + selected
        unique: list[str] = []
        seen: set[str] = set()
        for text in merged:
            key = _fingerprint(text)
            if key not in seen:
                seen.add(key)
                unique.append(text)
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            for text in unique:
                handle.write(json.dumps({"text": text, "source": "continual-learning"}, ensure_ascii=False) + "\n")
        fingerprint = hashlib.sha256("\n".join(_fingerprint(x) for x in unique).encode("utf-8")).hexdigest()
        return ReplayReport(
            input_records=len(base) + len(replay),
            eligible_records=len(replay),
            selected_records=len(selected),
            duplicates_removed=len(merged) - len(unique),
            output_path=str(destination),
            fingerprint=fingerprint,
        )


@dataclass(frozen=True)
class ConsolidationReport:
    """Result of a deterministic long-term-memory consolidation pass."""

    before: int
    after: int
    forgotten: tuple[str, ...]
    retained: tuple[str, ...]


class MemoryConsolidator:
    """Consolidate LongTermMemory without inventing or rewriting memories."""

    def consolidate(self, memory, max_records: int) -> ConsolidationReport:
        before = len(memory.keys())
        forgotten = tuple(memory.forget_least_used(max_records))
        retained = tuple(memory.keys())
        return ConsolidationReport(
            before=before,
            after=len(retained),
            forgotten=forgotten,
            retained=retained,
        )
