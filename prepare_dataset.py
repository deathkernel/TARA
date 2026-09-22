"""Prepare local or API-backed sources into a TARA JSONL corpus."""

from __future__ import annotations

import argparse

from src.dataset_pipeline import DatasetBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a clean TARA training dataset")
    parser.add_argument("sources", nargs="+", help="local JSON/JSONL/TXT files or HTTP(S) dataset endpoints")
    parser.add_argument("--output", default="data/tara_training.jsonl")
    parser.add_argument("--name", default="tara-training")
    parser.add_argument("--version", default="v1")
    args = parser.parse_args()

    builder = DatasetBuilder(args.name, args.version)
    records, manifest = builder.prepare(args.sources)
    builder.write_jsonl(records, manifest, args.output)
    print(f"Prepared {manifest.unique_records} unique records")
    print(f"Removed {manifest.duplicates_removed} duplicates")
    print(f"Fingerprint: {manifest.fingerprint}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
