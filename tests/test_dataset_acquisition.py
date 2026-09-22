from pathlib import Path

import pytest

from src.dataset_acquisition import (
    CuratedRecord,
    DatasetAcquisitionError,
    HuggingFaceStreamer,
    PublicDatasetAcquirer,
    PublicDatasetSpec,
    load_public_dataset_manifest,
    select_level,
)


def test_manifest_contains_basic_public_sources():
    specs = load_public_dataset_manifest("data/dataset_sources.json")
    basic = select_level(specs, "basic")
    keys = {spec.key for spec in basic}
    assert {"fineweb_edu", "gsm8k"}.issubset(keys)


def test_select_level_rejects_unknown_level():
    specs = [
        PublicDatasetSpec(
            key="x",
            name="X",
            organization="Test",
            dataset_id="test/x",
            config=None,
            split="train",
            text_field="text",
            level="basic",
            license="test",
            role="test",
        )
    ]
    with pytest.raises(DatasetAcquisitionError):
        select_level(specs, "advanced")


class FakeStreamer:
    def stream(self, spec, limit):
        for index in range(min(limit, 2)):
            yield CuratedRecord(
                text="same" if index == 1 and spec.key == "second" else f"{spec.key}-{index}",
                source=spec.dataset_id,
                record_id=str(index),
                metadata={"license": spec.license},
            )


def test_acquirer_deduplicates_and_writes_manifest(tmp_path: Path):
    specs = [
        PublicDatasetSpec(
            key="first",
            name="First",
            organization="Test",
            dataset_id="test/first",
            config=None,
            split="train",
            text_field="text",
            level="basic",
            license="MIT",
            role="test",
        ),
        PublicDatasetSpec(
            key="second",
            name="Second",
            organization="Test",
            dataset_id="test/second",
            config=None,
            split="train",
            text_field="text",
            level="basic",
            license="MIT",
            role="test",
        ),
    ]

    output = tmp_path / "basic.jsonl"
    report = PublicDatasetAcquirer(FakeStreamer()).acquire(
        specs,
        output,
        max_records_per_source=2,
        level="basic",
    )

    assert report.emitted_records == 4
    assert report.duplicates_removed == 0
    assert output.exists()
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_streamer_validates_bounds():
    with pytest.raises(ValueError):
        HuggingFaceStreamer(min_chars=10, max_chars=5)
