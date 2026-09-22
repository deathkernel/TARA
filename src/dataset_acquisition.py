"""Streaming acquisition and deterministic curation for public Hugging Face datasets.

The acquisition boundary intentionally streams data instead of downloading giant
corpora. A bounded sample is materialized locally with source, license and
record provenance attached to every example.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.dataset_registry import DatasetSpec


@dataclass(frozen=True)
class PublicDatasetSpec:
    key: str
    name: str
    organization: str
    dataset_id: str
    config: str | None
    split: str
    text_field: str
    level: str
    license: str
    role: str
    answer_field: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class CuratedRecord:
    text: str
    source: str
    record_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionReport:
    level: str
    requested_per_source: int
    emitted_records: int
    duplicates_removed: int
    filtered_records: int
    sources: tuple[str, ...]
    fingerprint: str
    output_path: str


class DatasetAcquisitionError(ValueError):
    """Raised for invalid source manifests or streamed dataset records."""


def _normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_public_dataset_manifest(path: str | Path) -> list[PublicDatasetSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("datasets") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise DatasetAcquisitionError("dataset source manifest must contain a datasets list")

    result: list[PublicDatasetSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise DatasetAcquisitionError("dataset manifest entries must be objects")
        required = ("key", "name", "organization", "dataset_id", "split", "text_field", "level", "license", "role")
        missing = [key for key in required if not item.get(key)]
        if missing:
            raise DatasetAcquisitionError(f"manifest entry missing fields: {', '.join(missing)}")
        result.append(
            PublicDatasetSpec(
                key=str(item["key"]),
                name=str(item["name"]),
                organization=str(item["organization"]),
                dataset_id=str(item["dataset_id"]),
                config=str(item["config"]) if item.get("config") else None,
                split=str(item["split"]),
                text_field=str(item["text_field"]),
                level=str(item["level"]),
                license=str(item["license"]),
                role=str(item["role"]),
                answer_field=str(item["answer_field"]) if item.get("answer_field") else None,
                notes=str(item.get("notes", "")),
            )
        )
    return result


def select_level(specs: Iterable[PublicDatasetSpec], level: str) -> list[PublicDatasetSpec]:
    key = level.strip().lower()
    selected = [spec for spec in specs if spec.level.lower() == key]
    if not selected:
        available = ", ".join(sorted({spec.level for spec in specs}))
        raise DatasetAcquisitionError(f"no datasets registered for level {key!r}; available: {available}")
    return selected


class HuggingFaceStreamer:
    """Stream bounded examples from a Hugging Face dataset."""

    def __init__(
        self,
        *,
        min_chars: int = 40,
        max_chars: int = 12000,
        seed: int = 42,
        shuffle_buffer: int = 1000,
    ) -> None:
        if min_chars < 0 or max_chars < min_chars:
            raise ValueError("invalid text length bounds")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    def stream(self, spec: PublicDatasetSpec, limit: int) -> Iterable[CuratedRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "public dataset acquisition requires the 'datasets' package"
            ) from exc

        kwargs: dict[str, Any] = {
            "split": spec.split,
            "streaming": True,
        }
        if spec.config:
            dataset = load_dataset(spec.dataset_id, spec.config, **kwargs)
        else:
            dataset = load_dataset(spec.dataset_id, **kwargs)

        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        emitted = 0
        for index, example in enumerate(dataset):
            if emitted >= limit:
                break
            if not isinstance(example, Mapping):
                continue
            text = _normalize_text(example.get(spec.text_field, ""))
            if spec.answer_field:
                answer = _normalize_text(example.get(spec.answer_field, ""))
                if answer:
                    text = f"Question: {text} Answer: {answer}"
            if len(text) < self.min_chars:
                continue
            text = text[: self.max_chars].strip()
            if not text:
                continue

            record_id = str(example.get("id", index))
            metadata = {
                "dataset_key": spec.key,
                "dataset_name": spec.name,
                "organization": spec.organization,
                "dataset_id": spec.dataset_id,
                "split": spec.split,
                "license": spec.license,
                "role": spec.role,
                "source_notes": spec.notes,
            }
            yield CuratedRecord(text=text, source=spec.dataset_id, record_id=record_id, metadata=metadata)
            emitted += 1


class PublicDatasetAcquirer:
    """Materialize a deterministic, deduplicated sample from selected public datasets."""

    def __init__(self, streamer: HuggingFaceStreamer | None = None) -> None:
        self.streamer = streamer or HuggingFaceStreamer()

    def acquire(
        self,
        specs: Iterable[PublicDatasetSpec],
        output: str | Path,
        *,
        max_records_per_source: int = 1000,
        level: str = "basic",
    ) -> AcquisitionReport:
        if max_records_per_source <= 0:
            raise ValueError("max_records_per_source must be positive")

        selected = [spec for spec in specs if spec.level.lower() == level.lower()]
        if not selected:
            raise DatasetAcquisitionError(f"no source datasets selected for level {level!r}")

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        written: list[CuratedRecord] = []
        duplicates = 0
        filtered = 0

        for spec in selected:
            before = len(written)
            for record in self.streamer.stream(spec, max_records_per_source):
                key = _fingerprint(record.text)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                written.append(record)
            filtered += max(0, max_records_per_source - (len(written) - before))

        with output_path.open("w", encoding="utf-8") as handle:
            for record in written:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        fingerprint = hashlib.sha256(
            "\n".join(_fingerprint(record.text) for record in written).encode("utf-8")
        ).hexdigest()
        report = AcquisitionReport(
            level=level,
            requested_per_source=max_records_per_source,
            emitted_records=len(written),
            duplicates_removed=duplicates,
            filtered_records=filtered,
            sources=tuple(spec.dataset_id for spec in selected),
            fingerprint=fingerprint,
            output_path=str(output_path),
        )
        output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report
