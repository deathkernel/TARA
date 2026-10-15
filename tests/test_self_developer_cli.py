import json

from src import self_developer_cli


def test_status_command_does_not_require_llm(monkeypatch, capsys):
    class FakeEngine:
        def status(self):
            return {"branch": "feature/test", "integration": "manual approval required"}

    monkeypatch.setattr(self_developer_cli, "_engine", lambda: FakeEngine())
    assert self_developer_cli.main(["status"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["branch"] == "feature/test"
    assert result["integration"] == "manual approval required"


def test_failed_report_returns_nonzero(monkeypatch, capsys):
    class FakeEngine:
        def latest_report(self):
            return {"status": "failed", "error": "test failure"}

    monkeypatch.setattr(self_developer_cli, "_engine", lambda: FakeEngine())
    assert self_developer_cli.main(["report"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
