from src.dataset_audit import DatasetAuditor


def test_dataset_audit_detects_duplicates_and_short_records():
    report = DatasetAuditor(min_length=8).audit([
        "learn sorting algorithms",
        "learn sorting algorithms",
        "x",
    ])
    assert report.duplicate_records == 1
    assert report.short_records == 1
    assert report.healthy


def test_dataset_audit_detects_split_leakage():
    record = {"text": "shared example"}
    report = DatasetAuditor().audit(
        [record],
        train=[record],
        validation=[record],
        test=[],
    )
    assert report.train_validation_overlap == 1
    assert not report.healthy
    assert any(issue.code == "split-overlap" for issue in report.issues)


def test_dataset_audit_is_deterministic():
    data = ["alpha example", "beta example"]
    first = DatasetAuditor().audit(data)
    second = DatasetAuditor().audit(data)
    assert first.fingerprint == second.fingerprint
