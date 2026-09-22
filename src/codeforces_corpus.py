"""Build model-ready text corpora from prepared Codeforces JSONL records.

The adapter deliberately does not invent solutions or code. Records with an
editorial become problem+editorial reasoning examples; records without one
remain problem-only examples. Official tests are included as verification
context when present.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _examples(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            result.append({"input": _text(item.get("input")), "output": _text(item.get("output"))})
    return result


def _tests(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            result.append({"input": _text(item.get("input")), "output": _text(item.get("output"))})
    return result


def _section(name: str, value: str) -> str:
    value = value.strip()
    return f"\n{name}:\n{value}\n" if value else ""


def record_to_text(record: dict[str, Any], *, include_tests: bool = True) -> str:
    title = _text(record.get("title"))
    description = _text(record.get("description"))
    if not title or not description:
        raise ValueError("Codeforces record requires title and description")

    parts = [f"Problem: {title}\n\n{description}"]
    parts.append(_section("Input format", _text(record.get("input_format"))))
    parts.append(_section("Output format", _text(record.get("output_format"))))
    parts.append(_section("Interaction format", _text(record.get("interaction_format"))))
    parts.append(_section("Note", _text(record.get("note"))))

    examples = _examples(record.get("examples"))
    if examples:
        rendered = []
        for index, example in enumerate(examples, 1):
            rendered.append(f"Example {index} input:\n{example['input']}\nExample {index} output:\n{example['output']}")
        parts.append("\nExamples:\n" + "\n\n".join(rendered) + "\n")

    tags = record.get("tags")
    if isinstance(tags, list) and tags:
        parts.append("\nTags: " + ", ".join(_text(tag) for tag in tags if _text(tag)) + "\n")

    editorial = _text(record.get("editorial"))
    if editorial:
        parts.append("\nEditorial / approach:\n" + editorial + "\n")

    if include_tests:
        tests = _tests(record.get("official_tests"))
        if tests:
            rendered = []
            for index, test in enumerate(tests, 1):
                rendered.append(f"Test {index} input:\n{test['input']}\nTest {index} output:\n{test['output']}")
            parts.append("\nOfficial verification tests:\n" + "\n\n".join(rendered) + "\n")

    return "".join(parts).strip()


def load_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"line {line_number}: record must be an object")
            records.append(item)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(source: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    records = load_records(source_path)
    records.sort(key=lambda item: _text(item.get("id")))

    problem_rows: list[dict[str, Any]] = []
    reasoning_rows: list[dict[str, Any]] = []
    for record in records:
        if not _text(record.get("id")) or not _text(record.get("title")) or not _text(record.get("description")):
            continue
        text = record_to_text(record)
        base = {"id": _text(record["id"]), "text": text}
        problem_rows.append(base)
        if _text(record.get("editorial")):
            reasoning_rows.append({**base, "has_editorial": True})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problem_path = out / "problem_corpus.jsonl"
    reasoning_path = out / "reasoning_corpus.jsonl"
    _write_jsonl(problem_path, problem_rows)
    _write_jsonl(reasoning_path, reasoning_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": source_path.as_posix(),
        "source_sha256": _fingerprint(source_path),
        "source_records": len(records),
        "problem_records": len(problem_rows),
        "reasoning_records": len(reasoning_rows),
        "reasoning_definition": "record has a non-empty editorial; no solution/code is invented",
        "outputs": {"problem": problem_path.name, "reasoning": reasoning_path.name},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="prepared Codeforces train.jsonl or validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("data/prepared/codeforces_corpus"))
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
