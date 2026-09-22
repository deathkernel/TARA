"""Basic dataset ingestion and preparation for TARA training.

Supports local JSON/JSONL/text files and HTTP(S) JSON/JSONL endpoints. The
pipeline normalizes records, removes duplicates, tracks provenance, and writes
a deterministic JSONL training corpus plus a manifest.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class DatasetRecord:
    """One normalized training example."""

    text: str
    source: str
    record_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetManifest:
    """Metadata describing a prepared dataset."""

    name: str
    version: str
    records: int
    unique_records: int
    duplicates_removed: int
    sources: tuple[str, ...]
    fingerprint: str


class DatasetError(ValueError):
    """Raised when a dataset cannot be parsed or validated."""


def _clean_text(value: Any) -> str:
    if not isinstance(value, str):
        raise DatasetError("record text must be a string")
    return re.sub(r"\s+", " ", value).strip()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _records_from_payload(payload: Any, source: str) -> list[DatasetRecord]:
    if isinstance(payload, dict):
        for key in ("records", "data", "items", "examples"):
            if key in payload:
                payload = payload[key]
                break
        else:
            payload = [payload]

    if not isinstance(payload, list):
        raise DatasetError("JSON dataset must be a list or contain records/data/items/examples")

    records: list[DatasetRecord] = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            text = _clean_text(item)
            metadata: dict[str, Any] = {}
            record_id = str(index)
        elif isinstance(item, dict):
            value = item.get("text")
            if value is None:
                value = item.get("content")
            if value is None:
                value = item.get("prompt")
            if value is None:
                raise DatasetError(f"record {index} has no text/content/prompt field")
            text = _clean_text(value)
            metadata = {
                str(k): v for k, v in item.items() if k not in {"text", "content", "prompt"}
            }
            record_id = str(item.get("id", index))
        else:
            raise DatasetError(f"record {index} must be a string or object")

        if text:
            records.append(DatasetRecord(text, source, record_id, metadata))
    return records


class DatasetSource:
    """Read training material from files or HTTP(S) endpoints."""

    @staticmethod
    def read(source: str | Path, timeout: float = 30.0) -> list[DatasetRecord]:
        source_text = str(source)
        if source_text.startswith(("http://", "https://")):
            request = Request(source_text, headers={"User-Agent": "TARA-dataset-pipeline/1.0"})
            with urlopen(request, timeout=timeout) as response:  # nosec B310 - scheme is explicitly restricted above
                raw = response.read().decode("utf-8")
            return DatasetSource._parse_text(raw, source_text)

        path = Path(source_text)
        if not path.exists():
            raise DatasetError(f"dataset source does not exist: {path}")
        return DatasetSource._parse_text(path.read_text(encoding="utf-8"), str(path), path.suffix.lower())

    @staticmethod
    def _parse_text(raw: str, source: str, suffix: str = "") -> list[DatasetRecord]:
        if suffix in {".txt", ".text"}:
            return [
                DatasetRecord(line.strip(), source, str(i))
                for i, line in enumerate(raw.splitlines())
                if line.strip()
            ]

        try:
            payload = json.loads(raw)
            return _records_from_payload(payload, source)
        except json.JSONDecodeError:
            records: list[DatasetRecord] = []
            for i, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"invalid JSONL at line {i + 1}: {exc}") from exc
                records.extend(_records_from_payload([payload], source))
            return records


class DatasetBuilder:
    """Combine, clean, deduplicate, split, and persist training data."""

    def __init__(self, name: str = "tara-training", version: str = "v1") -> None:
        self.name = name
        self.version = version

    def prepare(self, sources: Iterable[str | Path]) -> tuple[list[DatasetRecord], DatasetManifest]:
        collected: list[DatasetRecord] = []
        source_names: list[str] = []
        for source in sources:
            source_names.append(str(source))
            collected.extend(DatasetSource.read(source))

        unique: list[DatasetRecord] = []
        seen: set[str] = set()
        for record in collected:
            key = _fingerprint(record.text)
            if key in seen:
                continue
            seen.add(key)
            unique.append(record)

        dataset_hash = hashlib.sha256(
            "\n".join(_fingerprint(record.text) for record in unique).encode("utf-8")
        ).hexdigest()
        manifest = DatasetManifest(
            name=self.name,
            version=self.version,
            records=len(collected),
            unique_records=len(unique),
            duplicates_removed=len(collected) - len(unique),
            sources=tuple(source_names),
            fingerprint=dataset_hash,
        )
        return unique, manifest

    def write_jsonl(
        self,
        records: Iterable[DatasetRecord],
        manifest: DatasetManifest,
        output: str | Path,
    ) -> DatasetManifest:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        record_list = list(records)
        with path.open("w", encoding="utf-8") as handle:
            for record in record_list:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(asdict(manifest), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def split(
        records: list[DatasetRecord],
        validation_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ) -> tuple[list[DatasetRecord], list[DatasetRecord], list[DatasetRecord]]:
        if not 0 <= validation_ratio < 1 or not 0 <= test_ratio < 1:
            raise DatasetError("split ratios must be between 0 and 1")
        if validation_ratio + test_ratio >= 1:
            raise DatasetError("validation_ratio + test_ratio must be below 1")
        n = len(records)
        test_count = int(n * test_ratio)
        validation_count = int(n * validation_ratio)
        train_end = n - validation_count - test_count
        return records[:train_end], records[train_end : train_end + validation_count], records[-test_count:] if test_count else []
