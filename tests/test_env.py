from src.env import load_dotenv


def test_load_dotenv_reads_local_file_without_overriding_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('GROQ_API_KEY="test-key"\nOTHER=value\n', encoding="utf-8")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    load_dotenv(env_file)

    assert __import__("os").environ["GROQ_API_KEY"] == "test-key"
    assert __import__("os").environ["OTHER"] == "value"


def test_load_dotenv_does_not_override_environment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("GROQ_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "shell-key")

    load_dotenv(env_file)

    assert __import__("os").environ["GROQ_API_KEY"] == "shell-key"
