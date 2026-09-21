from pathlib import Path

import pytest

from src.checkpoint import load_checkpoint, save_checkpoint
from src.language_model import TinyLanguageModel
from src.optimizers import AdamW
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


def test_checkpoint_optimizer_round_trip(tmp_path):
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    optimizer = AdamW(model.parameters(), learning_rate=0.01, weight_decay=0.01)
    loss = model.loss([1, 2], [2, 3])
    loss.backward()
    optimizer.step()
    expected = optimizer.state_dict()
    path = tmp_path / "tara-v2.json"

    save_checkpoint(model, path, step=1, optimizer=optimizer)
    restored_model = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    restored_optimizer = AdamW(restored_model.parameters(), learning_rate=0.1)
    state = load_checkpoint(restored_model, path, optimizer=restored_optimizer)

    assert state["step"] == 1
    assert restored_optimizer.state_dict() == expected
    assert [p.data for p in restored_model.parameters()] == [p.data for p in model.parameters()]


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


def test_checkpoint_rejects_nonfinite_parameters(tmp_path):
    path = Path(tmp_path) / "bad.json"
    path.write_text('{"format_version": 1, "step": 0, "parameters": [NaN]}')
    model = TinyLanguageModel(vocab_size=5, embedding_dim=3, ff_dim=6, seed=4)
    with pytest.raises(ValueError):
        load_checkpoint(model, path)
