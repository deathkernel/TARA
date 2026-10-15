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
