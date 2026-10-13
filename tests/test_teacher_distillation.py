import json

import pytest

from src import teacher_distillation
from src.teacher_distillation import DOMAIN_TASKS, audit_teacher_dataset, generate_teacher_dataset


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
            (row["domain"] + "\n" + row["instruction"] + "\n" + row["answer"]).encode("utf-8")
        ).hexdigest()
        assert row["fingerprint"] == expected
        assert row["source"] == "ollama_teacher_generated"
        assert row["teacher_model"] == "fake-teacher"
        assert row["review_status"] == "unverified_candidate"
        assert row["text"] == (
            f"Domain: {row['domain']}\nInstruction: {row['instruction']}\nAnswer: {row['answer']}"
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
    output.write_text("known-good-data\n", encoding="utf-8")

    class FailingProvider:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("teacher unavailable")

    with pytest.raises(RuntimeError, match="teacher unavailable"):
        generate_teacher_dataset(
            FailingProvider(), output, per_domain=1, overwrite=True,
        )

    assert output.read_text(encoding="utf-8") == "known-good-data\n"
    assert not (tmp_path / "teacher.jsonl.tmp").exists()



def test_audit_accepts_generated_candidates_but_never_certifies_training_readiness(tmp_path):
    output = tmp_path / "teacher.jsonl"
    generate_teacher_dataset(FakeProvider(), output, per_domain=1)

    report = audit_teacher_dataset(output)

    assert report["records"] == len(DOMAIN_TASKS)
    assert report["valid_records"] == len(DOMAIN_TASKS)
    assert report["invalid_records"] == 0
    assert report["issues"] == []
    assert report["candidate_only"] is True
    assert report["semantic_review_required"] is True
    assert report["ready_for_training"] is False


def test_audit_reports_tampering_duplicate_and_blank_lines(tmp_path):
    output = tmp_path / "candidate.jsonl"
    provider = FakeProvider()
    generated = tmp_path / "generated.jsonl"
    generate_teacher_dataset(provider, generated, per_domain=1)
    first = json.loads(generated.read_text(encoding="utf-8").splitlines()[0])
    tampered = dict(first)
    tampered["answer"] = "Tampered answer content long enough to pass length checks."
    output.write_text(
        json.dumps(first) + "\n" + json.dumps(tampered) + "\n\n" + json.dumps(first) + "\n",
        encoding="utf-8",
    )

    report = audit_teacher_dataset(output)

    assert report["records"] == 4
    assert report["invalid_records"] == 3
    assert report["duplicate_fingerprints"] == 2
    assert any("fingerprint mismatch" in error for issue in report["issues"] for error in issue["errors"])
    assert any("blank line" in error for issue in report["issues"] for error in issue["errors"])
    assert any("duplicate fingerprint" in error for issue in report["issues"] for error in issue["errors"])
    assert report["ready_for_training"] is False


def test_audit_reports_invalid_json_and_malformed_domain_without_crashing(tmp_path):
    output = tmp_path / "malformed.jsonl"
    output.write_text('not-json\n' + json.dumps({"domain": [], "instruction": "", "answer": ""}) + "\n", encoding="utf-8")

    report = audit_teacher_dataset(output)

    assert report["records"] == 2
    assert report["invalid_records"] == 2
    assert len(report["issues"]) == 2
    assert any("invalid JSON" in error for issue in report["issues"] for error in issue["errors"])
    assert any("unknown or missing domain" in error for issue in report["issues"] for error in issue["errors"])


def test_audit_missing_file_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        audit_teacher_dataset(tmp_path / "missing.jsonl")



def test_audit_cli_runs_without_configuring_or_calling_teacher(monkeypatch, tmp_path, capsys):
    output = tmp_path / "teacher.jsonl"
    generate_teacher_dataset(FakeProvider(), output, per_domain=1)
    monkeypatch.setattr("sys.argv", ["teacher_distillation.py", "--audit", str(output)])

    with pytest.raises(SystemExit) as exit_info:
        teacher_distillation.main()

    assert exit_info.value.code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_records"] == 0
    assert report["ready_for_training"] is False


def test_audit_cli_returns_nonzero_for_invalid_dataset(monkeypatch, tmp_path, capsys):
    output = tmp_path / "bad.jsonl"
    output.write_text("not-json\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["teacher_distillation.py", "--audit", str(output)])

    with pytest.raises(SystemExit) as exit_info:
        teacher_distillation.main()

    assert exit_info.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_records"] == 1



def test_audit_reports_invalid_utf8_without_crashing(tmp_path):
    output = tmp_path / "invalid-utf8.jsonl"
    output.write_bytes(b'{"valid": false}\n' + b'\xff\xfe\n')

    report = audit_teacher_dataset(output)

    assert report["records"] == 2
    assert report["invalid_records"] == 2
    assert any(
        "line is not valid UTF-8" in error
        for issue in report["issues"]
        for error in issue["errors"]
    )
