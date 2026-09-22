"""Streaming acquisition and deterministic curation for public Hugging Face datasets.

The preferred path uses the dependency-light Hugging Face Dataset Viewer REST API,
so public parquet-backed datasets can be sampled without importing pandas/pyarrow.
A fallback to the optional datasets library is retained for environments where the
viewer cannot serve a dataset.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


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
        required = (
            "key",
            "name",
            "organization",
            "dataset_id",
            "split",
            "text_field",
            "level",
            "license",
            "role",
        )
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
        raise DatasetAcquisitionError(
            f"no datasets registered for level {key!r}; available: {available}"
        )
    return selected


class HuggingFaceStreamer:
    """Stream bounded examples without requiring pandas or pyarrow."""

    def __init__(
        self,
        *,
        min_chars: int = 40,
        max_chars: int = 12000,
        seed: int = 42,
        shuffle_buffer: int = 1000,
        api_base: str = "https://datasets-server.huggingface.co",
        allow_datasets_fallback: bool = False,
    ) -> None:
        if min_chars < 0 or max_chars < min_chars:
            raise ValueError("invalid text length bounds")
        if shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be positive")
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.api_base = api_base.rstrip("/")
        self.allow_datasets_fallback = allow_datasets_fallback
        self.last_filtered = 0

    def _get_json(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.api_base}/{endpoint}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "TARA-dataset-pipeline/2.0"})
        try:
            with urlopen(request, timeout=30.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DatasetAcquisitionError(
                f"Hugging Face Dataset Viewer request failed: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise DatasetAcquisitionError(
                "Hugging Face Dataset Viewer returned a non-object response"
            )
        if payload.get("error"):
            raise DatasetAcquisitionError(str(payload["error"]))
        return payload

    def _resolve_config(self, spec: PublicDatasetSpec) -> str:
        if spec.config:
            return spec.config
        payload = self._get_json("splits", {"dataset": spec.dataset_id})
        splits = payload.get("splits")
        if isinstance(splits, list):
            for item in splits:
                if isinstance(item, Mapping) and item.get("split") == spec.split and item.get("config"):
                    return str(item["config"])
        return "default"

    def _rows(self, spec: PublicDatasetSpec, config: str, offset: int, length: int = 100) -> dict[str, Any]:
        return self._get_json(
            "rows",
            {
                "dataset": spec.dataset_id,
                "config": config,
                "split": spec.split,
                "offset": offset,
                "length": min(100, max(1, length)),
            },
        )

    def _stream_viewer(self, spec: PublicDatasetSpec, limit: int) -> Iterable[CuratedRecord]:
        config = self._resolve_config(spec)
        first = self._rows(spec, config, 0, 1)
        total = int(first.get("num_rows_total") or 0)
        if total <= 0:
            return

        digest = hashlib.sha256(f"{spec.dataset_id}:{self.seed}".encode("utf-8")).digest()
        offset = int.from_bytes(digest[:8], "big") % total
        visited = 0
        emitted = 0

        while visited < total and emitted < limit:
            length = min(100, total - visited, limit - emitted + 100)
            payload = self._rows(spec, config, offset, length)
            rows = payload.get("rows")
            if not isinstance(rows, list) or not rows:
                break

            for entry in rows:
                if emitted >= limit:
                    break
                if not isinstance(entry, Mapping):
                    self.last_filtered += 1
                    continue
                example = entry.get("row")
                if not isinstance(example, Mapping):
                    self.last_filtered += 1
                    continue

                text = _normalize_text(example.get(spec.text_field, ""))
                if spec.answer_field:
                    answer = _normalize_text(example.get(spec.answer_field, ""))
                    if answer:
                        text = f"Question: {text} Answer: {answer}"

                if len(text) < self.min_chars:
                    self.last_filtered += 1
                    continue

                text = text[: self.max_chars].strip()
                if not text:
                    self.last_filtered += 1
                    continue

                row_id = entry.get("row_idx")
                record_id = str(row_id if row_id is not None else emitted)
                metadata = {
                    "dataset_key": spec.key,
                    "dataset_name": spec.name,
                    "organization": spec.organization,
                    "dataset_id": spec.dataset_id,
                    "config": config,
                    "split": spec.split,
                    "license": spec.license,
                    "role": spec.role,
                    "source_notes": spec.notes,
                    "acquisition": "huggingface-dataset-viewer-rows",
                }
                yield CuratedRecord(
                    text=text,
                    source=spec.dataset_id,
                    record_id=record_id,
                    metadata=metadata,
                )
                emitted += 1

            consumed = len(rows)
            visited += consumed
            offset = (offset + consumed) % total

    def _stream_datasets(self, spec: PublicDatasetSpec, limit: int) -> Iterable[CuratedRecord]:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DatasetAcquisitionError(
                "Hugging Face Dataset Viewer failed and optional 'datasets' fallback is unavailable"
            ) from exc

        kwargs: dict[str, Any] = {"split": spec.split, "streaming": True}
        dataset = (
            load_dataset(spec.dataset_id, spec.config, **kwargs)
            if spec.config
            else load_dataset(spec.dataset_id, **kwargs)
        )
        if hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)

        emitted = 0
        for index, example in enumerate(dataset):
            if emitted >= limit:
                break
            if not isinstance(example, Mapping):
                self.last_filtered += 1
                continue
            text = _normalize_text(example.get(spec.text_field, ""))
            if spec.answer_field:
                answer = _normalize_text(example.get(spec.answer_field, ""))
                if answer:
                    text = f"Question: {text} Answer: {answer}"
            if len(text) < self.min_chars:
                self.last_filtered += 1
                continue
            text = text[: self.max_chars].strip()
            if not text:
                self.last_filtered += 1
                continue
            metadata = {
                "dataset_key": spec.key,
                "dataset_name": spec.name,
                "organization": spec.organization,
                "dataset_id": spec.dataset_id,
                "config": spec.config,
                "split": spec.split,
                "license": spec.license,
                "role": spec.role,
                "source_notes": spec.notes,
                "acquisition": "huggingface-datasets-streaming",
            }
            yield CuratedRecord(
                text=text,
                source=spec.dataset_id,
                record_id=str(example.get("id", index)),
                metadata=metadata,
            )
            emitted += 1

    def stream(self, spec: PublicDatasetSpec, limit: int) -> Iterable[CuratedRecord]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.last_filtered = 0
        try:
            yield from self._stream_viewer(spec, limit)
        except DatasetAcquisitionError:
            if not self.allow_datasets_fallback:
                raise
            yield from self._stream_datasets(spec, limit)


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

        selected = select_level(specs, level)

        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seen: set[str] = set()
        written: list[CuratedRecord] = []
        duplicates = 0
        filtered = 0

        for spec in selected:
            for record in self.streamer.stream(spec, max_records_per_source):
                key = _fingerprint(record.text)
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                written.append(record)
            filtered += getattr(self.streamer, "last_filtered", 0)

        if not written:
            raise DatasetAcquisitionError(
                "no usable records were produced; check dataset availability, source configuration, or filters"
            )

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
