from pathlib import Path

import pytest

from src.dataset_acquisition import (
    CuratedRecord,
    DatasetAcquisitionError,
    _normalize_dialogue,
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
            if spec.key == "second" and index == 1:
                text = "first-0"
            else:
                text = f"{spec.key}-{index}"
            yield CuratedRecord(
                text=text,
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

    assert report.emitted_records == 3
    assert report.duplicates_removed == 1
    assert output.exists()
    assert output.with_suffix(".jsonl.manifest.json").exists()


def test_streamer_validates_bounds():
    with pytest.raises(ValueError):
        HuggingFaceStreamer(min_chars=10, max_chars=5)


def test_streamer_does_not_need_datasets_import(monkeypatch):
    streamer = HuggingFaceStreamer()
    calls = []

    def fake_get_json(endpoint, params):
        calls.append((endpoint, params))
        if endpoint == "splits":
            return {"splits": [{"dataset": "demo/source", "config": "default", "split": "train"}]}
        return {
            "rows": [
                {"row_idx": 7, "row": {"text": "This is a valid public dataset example."}}
            ],
            "num_rows_total": 8,
        }

    monkeypatch.setattr(streamer, "_get_json", fake_get_json)
    spec = PublicDatasetSpec(
        key="demo",
        name="Demo",
        organization="Test",
        dataset_id="demo/source",
        config=None,
        split="train",
        text_field="text",
        level="basic",
        license="MIT",
        role="test",
    )

    records = list(streamer.stream(spec, 1))
    assert len(records) == 1
    assert records[0].text.startswith("This is a valid public")
    assert any(endpoint == "rows" for endpoint, _ in calls)


def test_manifest_contains_conversation_curriculum():
    specs = load_public_dataset_manifest("data/dataset_sources.json")
    conversation = select_level(specs, "conversation-v1")
    assert len(conversation) == 1
    assert conversation[0].key == "soda"
    assert conversation[0].dialogue_field == "dialogue"
    assert conversation[0].speaker_field == "speakers"


def test_normalize_dialogue_uses_stable_roles():
    text = _normalize_dialogue(
        ["Hi there", "Hey!", "What are you building?", "A little AI project."],
        ["Alice", "Bob", "Alice", "Bob"],
    )
    assert text == (
        "<|user|>\nHi there\n"
        "<|assistant|>\nHey!\n"
        "<|user|>\nWhat are you building?\n"
        "<|assistant|>\nA little AI project."
    )


def test_normalize_dialogue_rejects_more_than_two_speakers():
    assert _normalize_dialogue(["a", "b", "c"], ["A", "B", "C"]) == ""
