import json

import pytest

from src.teacher_distillation import DOMAIN_TASKS, generate_teacher_dataset


class FakeProvider:
    model = "fake-teacher"

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return f"Verified-format candidate answer for: {prompt}"


def test_generates_examples_across_all_domains(tmp_path):
    provider = FakeProvider()
    output = tmp_path / "teacher.jsonl"

    counts = generate_teacher_dataset(provider, output, per_domain=2)

    assert counts["total"] == len(DOMAIN_TASKS) * 2
    assert set(counts) == {*DOMAIN_TASKS, "total"}
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == counts["total"]
    assert {row["domain"] for row in rows} == set(DOMAIN_TASKS)
    assert all(row["review_status"] == "unverified_candidate" for row in rows)
    assert all(row["text"].startswith("Domain: ") for row in rows)
    assert all(kwargs["temperature"] == 0.2 for _, kwargs in provider.prompts)


def test_refuses_to_overwrite_existing_dataset(tmp_path):
    output = tmp_path / "existing.jsonl"
    output.write_text("keep me\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        generate_teacher_dataset(FakeProvider(), output, per_domain=1)

    assert output.read_text(encoding="utf-8") == "keep me\n"


def test_invalid_generation_settings_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        generate_teacher_dataset(FakeProvider(), tmp_path / "out.jsonl", per_domain=0)


def test_candidate_fingerprint_and_provenance_are_reproducible(tmp_path):
    import hashlib

    provider = FakeProvider()
    output = tmp_path / "teacher.jsonl"
    generate_teacher_dataset(provider, output, per_domain=1)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    for row in rows:
        expected = hashlib.sha256(
            (row["domain"] + "\\n" + row["instruction"] + "\\n" + row["answer"]).encode("utf-8")
        ).hexdigest()
        assert row["fingerprint"] == expected
        assert row["source"] == "ollama_teacher_generated"
        assert row["teacher_model"] == "fake-teacher"
        assert row["review_status"] == "unverified_candidate"
        assert row["text"] == (
            f"Domain: {row['domain']}\\nInstruction: {row['instruction']}\\nAnswer: {row['answer']}"
        )


def test_short_teacher_answers_are_skipped_without_bad_records(tmp_path):
    class ShortAnswerProvider:
        model = "short-teacher"

        def complete(self, prompt, **kwargs):
            return "  too short  "

    output = tmp_path / "teacher.jsonl"
    counts = generate_teacher_dataset(ShortAnswerProvider(), output, per_domain=1)

    assert counts["total"] == 0
    assert all(counts[domain] == 0 for domain in DOMAIN_TASKS)
    assert output.read_text(encoding="utf-8") == ""


def test_provider_failure_does_not_replace_existing_dataset_or_leave_temp(tmp_path):
    output = tmp_path / "teacher.jsonl"
    output.write_text("known-good-data\\n", encoding="utf-8")

    class FailingProvider:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("teacher unavailable")

    with pytest.raises(RuntimeError, match="teacher unavailable"):
        generate_teacher_dataset(
            FailingProvider(), output, per_domain=1, overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "known-good-data\\n"
    assert not (tmp_path / "teacher.jsonl.tmp").exists()
