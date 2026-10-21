from pathlib import Path

import pytest

from src.polyglot.model_adapter import TARAAlgorithmModel


def test_model_adapter_reports_missing_checkpoint(tmp_path: Path):
    model = TARAAlgorithmModel(tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="checkpoint not found"):
        model.load()


def test_model_adapter_lazy_loads_and_generates(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")

    fake_model = object()
    fake_tokenizer = object()
    seen = {}

    def fake_load(path):
        seen["path"] = path
        return fake_model, fake_tokenizer

    def fake_generate(model, tokenizer, prompt, max_new_tokens, temperature):
        assert model is fake_model
        assert tokenizer is fake_tokenizer
        return f"{prompt}:{max_new_tokens}:{temperature}"

    monkeypatch.setattr("src.model_runtime.load_checkpoint", fake_load)
    monkeypatch.setattr("src.model_runtime.generate_text", fake_generate)

    runtime = TARAAlgorithmModel(checkpoint, max_new_tokens=123)
    assert not runtime.loaded
    output = runtime.generate("hello", temperature=0.7)
    assert runtime.loaded
    assert seen["path"] == checkpoint
    assert output.endswith(":123:0.7")


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "12"])
def test_model_adapter_rejects_invalid_default_token_limit(tmp_path: Path, value):
    with pytest.raises(ValueError, match="positive integer"):
        TARAAlgorithmModel(tmp_path / "model.pt", max_new_tokens=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "12"])
def test_model_adapter_rejects_invalid_generation_token_limit(monkeypatch, tmp_path: Path, value):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr("src.model_runtime.load_checkpoint", lambda path: (object(), object()))
    model = TARAAlgorithmModel(checkpoint)
    with pytest.raises(ValueError, match="positive integer"):
        model.generate("hello", max_new_tokens=value)
