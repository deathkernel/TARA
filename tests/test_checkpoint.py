from pathlib import Path

import pytest

from src.checkpoint import load_checkpoint, save_checkpoint
from src.language_model import TinyLanguageModel
from src.schedulers import CosineAnnealing


def test_checkpoint_round_trip(tmp_path):
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    original = [parameter.data for parameter in model.parameters()]
    scheduler = CosineAnnealing(0.03, total_steps=10, min_lr=0.003)
    path = tmp_path / "tara.json"

    save_checkpoint(
        model, path, step=7, scheduler=scheduler, metrics={"val_loss": 1.25}
    )

    for parameter in model.parameters():
        parameter.data += 10.0

    state = load_checkpoint(model, path)
    assert [parameter.data for parameter in model.parameters()] == original
    assert state["step"] == 7
    assert state["metrics"]["val_loss"] == 1.25
    assert state["scheduler"]["initial_lr"] == 0.03


def test_checkpoint_rejects_model_size_mismatch(tmp_path):
    path = tmp_path / "tara.json"
    model_a = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    save_checkpoint(model_a, path, step=1)
    model_b = TinyLanguageModel(vocab_size=6, embedding_dim=3, ff_dim=6, seed=4)
    with pytest.raises(ValueError):
        load_checkpoint(model_b, path)


def test_checkpoint_rejects_unknown_version(tmp_path):
    path = Path(tmp_path) / "bad.json"
    path.write_text('{"format_version": 99, "step": 0, "parameters": []}')
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    with pytest.raises(ValueError):
        load_checkpoint(model, path)
