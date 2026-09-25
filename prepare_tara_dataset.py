"""Prepare a bounded public TARA dataset sample from the source manifest."""

from __future__ import annotations

import argparse

from src.dataset_acquisition import (
    PublicDatasetAcquirer,
    load_public_dataset_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a bounded TARA dataset sample")
    parser.add_argument("--manifest", default="data/dataset_sources.json")
    parser.add_argument("--level", default="basic", choices=("basic", "conversation-v1", "intermediate", "advanced", "specialized"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-records-per-source", type=int, default=1000)
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--max-chars", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    args = parser.parse_args()

    output = args.output or f"data/{args.level}_v1.jsonl"
    specs = load_public_dataset_manifest(args.manifest)
    from src.dataset_acquisition import HuggingFaceStreamer

    streamer = HuggingFaceStreamer(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    report = PublicDatasetAcquirer(streamer).acquire(
        specs,
        output,
        max_records_per_source=args.max_records_per_source,
        level=args.level,
    )
    print(
        f"level={report.level} emitted={report.emitted_records} "
        f"duplicates_removed={report.duplicates_removed} "
        f"filtered={report.filtered_records} output={report.output_path} "
        f"fingerprint={report.fingerprint}"
    )


if __name__ == "__main__":
    main()
