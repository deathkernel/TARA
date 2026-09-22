"""Deterministic mixer for base and failure-targeted training datasets.

Targeted examples are synthetic supervision derived from benchmark failures.
They are intentionally kept separate from verified replay knowledge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MixReport:
    base_records: int
    targeted_records: int
    selected_targeted: int
    duplicates_removed: int
    output_path: str
    fingerprint: str


def _record_text(item: object) -> str:
    if isinstance(item, str):
        return item.strip()
    if not isinstance(item, dict):
        raise ValueError("dataset record must be a string or object")
    if "problem" in item and "solution" in item:
        return (
            "Problem: " + str(item["problem"])
            + "\nApproach: " + str(item["solution"])
            + "\nTests: " + str(item.get("tests", ""))
        ).strip()
    if "prompt" in item and "solution" in item:
        return (
            "Prompt: " + str(item["prompt"])
            + "\nAnswer: " + str(item["solution"])
            + "\nTests: " + str(item.get("tests", ""))
        ).strip()
    value = item.get("text", item.get("content", item.get("prompt")))
    if value is None:
        raise ValueError("dataset record has no supported text fields")
    return str(value).strip()


def _load(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    raw = source.read_text(encoding="utf-8")
    if source.suffix.lower() in {".txt", ".text"}:
        values = [line.strip() for line in raw.splitlines() if line.strip()]
    else:
        lines = [line for line in raw.splitlines() if line.strip()]
        try:
            payload = json.loads(raw)
            items = payload.get("records", payload) if isinstance(payload, dict) else payload
            items = items if isinstance(items, list) else [items]
        except json.JSONDecodeError:
            items = [json.loads(line) for line in lines]
        values = [_record_text(item) for item in items]
    return [value for value in values if value]


class TargetedDatasetMixer:
    """Merge base data with a bounded fraction of targeted examples."""

    def __init__(self, targeted_ratio: float = 0.5, seed: int = 42) -> None:
        if not 0.0 <= targeted_ratio <= 1.0:
            raise ValueError("targeted_ratio must be in [0, 1]")
        self.targeted_ratio = targeted_ratio
        self.seed = seed

    def mix(self, base_path: str | Path, targeted_path: str | Path, output_path: str | Path) -> MixReport:
        base = _load(base_path)
        targeted = _load(targeted_path)
        if not base:
            raise ValueError("base dataset is empty")
        if not targeted:
            raise ValueError("targeted dataset is empty")

        seen: set[str] = set()
        base_unique: list[str] = []
        for text in base:
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint not in seen:
                seen.add(fingerprint)
                base_unique.append(text)

        targeted_unique: list[str] = []
        duplicates = len(base) - len(base_unique)
        for text in targeted:
            fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if fingerprint in seen:
                duplicates += 1
                continue
            seen.add(fingerprint)
            targeted_unique.append(text)

        if self.targeted_ratio == 0.0:
            selected_count = 0
        else:
            selected_count = min(
                len(targeted_unique),
                max(1, round(len(base_unique) * self.targeted_ratio / (1.0 - self.targeted_ratio)))
                if self.targeted_ratio < 1.0
                else len(targeted_unique),
            )
        selected = targeted_unique[:selected_count]
        merged = base_unique + selected
        fingerprint = hashlib.sha256("\n".join(merged).encode("utf-8")).hexdigest()

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for text in merged:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        return MixReport(len(base), len(targeted), len(selected), duplicates, str(output), fingerprint)
