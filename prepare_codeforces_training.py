"""Prepare Codeforces Parquet data for TARA training.

Reads Codeforces problems from Parquet without loading the whole dataset into
pandas, normalizes useful fields, creates a deterministic train/validation
split, and writes JSONL plus a manifest containing a source fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCHEMA_VERSION = 1
DEFAULT_COLUMNS = [
    "id", "title", "description", "input_format", "output_format",
    "interaction_format", "note", "examples", "editorial", "rating",
    "tags", "official_tests", "official_tests_complete", "executable",
]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    try:
        return value.as_py()
    except AttributeError:
        return str(value)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_content(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return bool(str(value).strip())


def _fingerprint(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    examples = _jsonable(row.get("examples")) or []
    tags = _jsonable(row.get("tags")) or []
    record = {
        "id": _text(row.get("id")),
        "title": _text(row.get("title")),
        "description": _text(row.get("description")),
        "input_format": _text(row.get("input_format")),
        "output_format": _text(row.get("output_format")),
        "interaction_format": _text(row.get("interaction_format")),
        "note": _text(row.get("note")),
        "examples": examples if isinstance(examples, list) else [],
        "editorial": _text(row.get("editorial")),
        "rating": row.get("rating"),
        "tags": tags if isinstance(tags, list) else [],
        "official_tests": _jsonable(row.get("official_tests")),
        "official_tests_complete": bool(row.get("official_tests_complete")),
        "executable": _text(row.get("executable")),
    }
    record["signals"] = {
        "has_editorial": bool(record["editorial"]),
        "has_examples": bool(record["examples"]),
        "has_official_tests": _has_content(record["official_tests"]),
        "has_executable": bool(record["executable"]),
    }
    return record


def prepare(source: Path, output_dir: Path, validation_ratio: float, seed: int) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between 0 and 1")

    parquet = pq.ParquetFile(source)
    available = set(parquet.schema_arrow.names)
    columns = [c for c in DEFAULT_COLUMNS if c in available]
    rows = [normalize(r) for r in pq.read_table(source, columns=columns, use_threads=False).to_pylist()]
    rows = [r for r in rows if r["id"] and r["description"]]

    # Deterministic hash split: no dependence on row order or Python hash randomization.
    def bucket(record: dict[str, Any]) -> int:
        token = f"{seed}:{record['id']}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(token).digest()[:8], "big")

    rows.sort(key=lambda r: r["id"])
    validation_count = max(1, round(len(rows) * validation_ratio)) if rows else 0
    ranked = sorted(rows, key=bucket)
    validation_ids = {r["id"] for r in ranked[:validation_count]}
    train_rows = [r for r in rows if r["id"] not in validation_ids]
    validation_rows = [r for r in rows if r["id"] in validation_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    for path, data in ((train_path, train_rows), (validation_path, validation_rows)):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in data:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    source_sha256 = _fingerprint(source)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source.as_posix()),
        "source_sha256": source_sha256,
        "source_rows": parquet.metadata.num_rows,
        "usable_rows": len(rows),
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_ratio": validation_ratio,
        "seed": seed,
        "columns": columns,
        "outputs": {"train": train_path.name, "validation": validation_path.name},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/codeforces"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir, args.validation_ratio, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
