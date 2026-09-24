from src.env import load_dotenv


def test_load_dotenv_reads_local_file_without_overriding_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=value\n", encoding="utf-8")
    load_dotenv(env_file)
    assert __import__("os").environ["OTHER"] == "value"


def test_load_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=file-value\n", encoding="utf-8")
    monkeypatch.setenv("OTHER", "shell-value")
    load_dotenv(env_file)
    assert __import__("os").environ["OTHER"] == "shell-value"
