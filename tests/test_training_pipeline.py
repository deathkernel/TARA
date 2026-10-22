from src.checkpoint import load_checkpoint
from src.language_dataset import CausalTextDataset
from src.language_model import TinyLanguageModel
from train_tiny_lm import _batch_loss, _mean_loss


def test_batch_loss_is_token_weighted():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    batch = [([0, 1], [1, 2]), ([2], [3])]
    expected = (model.loss([0, 1], [1, 2]).data * 2 + model.loss([2], [3]).data) / 3
    assert abs(_batch_loss(model, batch).data - expected) < 1e-12


def test_validation_loss_does_not_change_parameters():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    dataset = CausalTextDataset([0, 1, 2, 3], context_length=2)
    before = [parameter.data for parameter in model.parameters()]
    value = _mean_loss(model, dataset, batch_size=2)
    after = [parameter.data for parameter in model.parameters()]
    assert value > 0.0
    assert after == before


def test_training_checkpoint_contains_final_metadata(tmp_path):
    from train_tiny_lm import train
    from src.tokenizer import CharTokenizer

    path = tmp_path / "final.json"
    train(steps=1, checkpoint_path=path)
    tokenizer = CharTokenizer("tara learns. tara reasons. ")
    state_model = TinyLanguageModel(vocab_size=tokenizer.vocab_size, embedding_dim=3, ff_dim=6, seed=7)
    state = load_checkpoint(state_model, path)
    assert state["step"] == 0
    assert "val_loss" in state["metrics"]
    assert state["scheduler"]["initial_lr"] == 0.03


import json

import pytest

torch = pytest.importorskip("torch")

from src.training_pipeline import TrainingConfig, TrainingPipeline, load_training_texts, split_texts


def write_algorithm_dataset(path, count=8):
    rows = [{"problem": f"sort list {i}", "solution": "return sorted(value)", "tests": "[3,1,2] -> [1,2,3]"} for i in range(count)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_phase14_loads_algorithm_records(tmp_path):
    path = tmp_path / "data.jsonl"
    write_algorithm_dataset(path, count=2)
    texts = load_training_texts(path)
    assert len(texts) == 2
    assert texts[0].startswith("Problem: sort list")
    assert "Approach: return sorted(value)" in texts[0]


def test_targeted_prompt_records_include_answer_and_tests(tmp_path):
    path = tmp_path / "targeted.jsonl"
    path.write_text(json.dumps({"prompt": "Compute 2 + 2", "solution": 4, "tests": "expected=4"}) + "\n", encoding="utf-8")
    text = load_training_texts(path)[0]
    assert "Prompt: Compute 2 + 2" in text
    assert "Answer: 4" in text
    assert "Tests: expected=4" in text


def test_phase14_split_is_deterministic_and_non_overlapping():
    texts = [f"record-{i}" for i in range(10)]
    first = split_texts(texts, 0.2, 7)
    second = split_texts(texts, 0.2, 7)
    assert first == second
    assert not set(first[0]) & set(first[1])
    assert len(first[1]) == 2


def tiny_config(steps=2):
    return TrainingConfig(steps=steps, batch_size=2, context=16, embedding_dim=8, ff_dim=16, heads=2, lr=1e-3, validation_split=0.25, seed=11, log_every=2)


def test_phase14_writes_self_contained_checkpoint(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    summary = TrainingPipeline(tiny_config()).train(data, checkpoint)
    assert summary.start_step == 0
    assert summary.final_step == 2
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["format_version"] == 3
    assert payload["step"] == 2
    assert "model_state" in payload
    assert "optimizer_state" in payload
    assert "tokenizer" in payload
    assert "dataset_fingerprint" in payload
    assert "hardening_config" in payload
    assert "scheduler" in payload
    assert "early_stopping" in payload
    assert "metrics_fingerprint" in payload
    assert payload["metrics"]["train_loss"] == pytest.approx(summary.train_loss)


def test_phase14_resume_continues_step_and_rejects_changed_dataset(tmp_path):
    data = tmp_path / "data.jsonl"
    changed = tmp_path / "changed.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    write_algorithm_dataset(changed, count=9)
    TrainingPipeline(tiny_config(steps=1)).train(data, checkpoint)
    resumed = TrainingPipeline(tiny_config(steps=2)).train(data, checkpoint, resume=checkpoint)
    assert resumed.start_step == 1
    assert resumed.final_step == 3
    with pytest.raises(ValueError, match="dataset fingerprint differs"):
        TrainingPipeline(tiny_config(steps=1)).train(changed, checkpoint, resume=checkpoint)


def test_phase14_config_rejects_invalid_attention_shape():
    with pytest.raises(ValueError, match="divisible"):
        TrainingConfig(embedding_dim=7, heads=2)


def test_phase37_gradient_accumulation_scheduler_and_tracker(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.jsonl"
    write_algorithm_dataset(data, count=8)
    config = TrainingConfig(**{**tiny_config(steps=3).__dict__, "gradient_accumulation_steps": 2, "warmup_steps": 1, "min_lr_ratio": 0.2})
    summary = TrainingPipeline(config).train(data, checkpoint, metrics_path=metrics)
    assert summary.final_step == 3
    rows = metrics.read_text(encoding="utf-8").splitlines()
    assert rows
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["hardening_config"]["gradient_accumulation_steps"] == 2
    assert payload["scheduler"]["optimizer_steps"] == 3
    assert payload["metrics_fingerprint"]


def test_training_preserves_best_validation_checkpoint(tmp_path, monkeypatch):
    import src.training_pipeline as training_pipeline_module

    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    validation_losses = iter([1.0, 0.5, 0.7, 0.9])
    monkeypatch.setattr(
        training_pipeline_module,
        "evaluate",
        lambda *args, **kwargs: next(validation_losses),
    )
    config = TrainingConfig(
        steps=4,
        batch_size=2,
        context=16,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        lr=1e-3,
        validation_split=0.25,
        seed=11,
        log_every=1,
        early_stopping_patience=10,
    )

    summary = TrainingPipeline(config).train(data, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert summary.final_step == 4
    assert payload["step"] == 2
    assert payload["metrics"]["validation_loss"] == pytest.approx(0.5)
    assert not checkpoint.with_name(checkpoint.name + ".best").exists()



def test_atomic_torch_save_replaces_checkpoint_with_complete_payload(tmp_path):
    from src.training_pipeline import _atomic_torch_save

    destination = tmp_path / "model.pt"
    destination.write_bytes(b"old-checkpoint")

    _atomic_torch_save({"step": 7, "weights": [1, 2, 3]}, destination)

    payload = torch.load(destination, map_location="cpu", weights_only=True)
    assert payload == {"step": 7, "weights": [1, 2, 3]}
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_atomic_torch_save_failure_preserves_existing_checkpoint(tmp_path, monkeypatch):
    import src.training_pipeline as training_pipeline_module

    destination = tmp_path / "model.pt"
    known_good = b"known-good-checkpoint-bytes"
    destination.write_bytes(known_good)

    def fail_save(payload, path):
        with open(path, "wb") as handle:
            handle.write(b"partial-checkpoint")
        raise OSError("simulated checkpoint write failure")

    monkeypatch.setattr(training_pipeline_module.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated checkpoint write failure"):
        training_pipeline_module._atomic_torch_save({"step": 8}, destination)

    assert destination.read_bytes() == known_good
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_atomic_torch_save_replace_failure_preserves_existing_checkpoint(tmp_path, monkeypatch):
    import src.training_pipeline as training_pipeline_module

    destination = tmp_path / "model.pt"
    known_good = b"known-good-checkpoint-bytes"
    destination.write_bytes(known_good)

    def fail_replace(source, target):
        raise OSError("simulated atomic replace failure")

    monkeypatch.setattr(training_pipeline_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated atomic replace failure"):
        training_pipeline_module._atomic_torch_save({"step": 9}, destination)

    assert destination.read_bytes() == known_good
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_resume_preserves_incumbent_best_validation_checkpoint(tmp_path, monkeypatch):
    import src.training_pipeline as training_pipeline_module

    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    validation_losses = iter([1.0, 0.5, 0.7, 0.9])
    monkeypatch.setattr(
        training_pipeline_module,
        "evaluate",
        lambda *args, **kwargs: next(validation_losses),
    )
    common = dict(
        batch_size=2,
        context=16,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        lr=1e-3,
        validation_split=0.25,
        seed=11,
        log_every=1,
        early_stopping_patience=10,
    )

    TrainingPipeline(TrainingConfig(steps=2, **common)).train(data, checkpoint)
    resumed = TrainingPipeline(TrainingConfig(steps=2, **common)).train(
        data, checkpoint, resume=checkpoint
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert resumed.start_step == 2
    assert resumed.final_step == 4
    assert payload["step"] == 2
    assert payload["metrics"]["validation_loss"] == pytest.approx(0.5)
    assert not checkpoint.with_name(checkpoint.name + ".best").exists()


def test_training_rng_state_round_trip_restores_python_and_torch(tmp_path):
    import random
    from src.training_pipeline import TrainingPipeline

    random.seed(1234)
    torch.manual_seed(5678)
    state = TrainingPipeline._capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(4)

    random.random()
    torch.rand(9)
    TrainingPipeline._restore_rng_state(state)

    assert random.random() == expected_python
    assert torch.equal(torch.rand(4), expected_torch)


def test_training_checkpoint_contains_rng_state(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)

    TrainingPipeline(tiny_config(steps=1)).train(data, checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert isinstance(payload.get("rng_state"), dict)
    assert isinstance(payload["rng_state"].get("python"), tuple)
    assert isinstance(payload["rng_state"].get("torch"), torch.Tensor)

def test_resume_from_best_checkpoint_path_preserves_source_until_publish(tmp_path, monkeypatch):
    import shutil
    import src.training_pipeline as training_pipeline_module

    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    best_checkpoint = checkpoint.with_name(checkpoint.name + ".best")
    write_algorithm_dataset(data, count=8)
    TrainingPipeline(tiny_config(steps=1)).train(data, checkpoint)
    shutil.copy2(checkpoint, best_checkpoint)
    expected_step = torch.load(best_checkpoint, map_location="cpu", weights_only=True)["step"]
    monkeypatch.setattr(training_pipeline_module, "evaluate", lambda *args, **kwargs: 100.0)

    summary = TrainingPipeline(tiny_config(steps=1)).train(
        data, checkpoint, resume=best_checkpoint
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert summary.start_step == expected_step
    assert summary.final_step == expected_step + 1
    assert payload["step"] == expected_step
    assert not best_checkpoint.exists()

def test_resume_restores_and_advances_optimizer_state(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    config = TrainingConfig(
        steps=1,
        batch_size=2,
        context=16,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        lr=1e-3,
        validation_split=0.0,
        seed=11,
        log_every=1,
    )

    TrainingPipeline(config).train(data, checkpoint)
    before = torch.load(checkpoint, map_location="cpu", weights_only=True)
    before_steps = [
        int(state["step"].item())
        for state in before["optimizer_state"]["state"].values()
        if "step" in state
    ]
    assert before_steps and set(before_steps) == {1}

    resumed_config = TrainingConfig(**{**config.__dict__, "steps": 1})
    summary = TrainingPipeline(resumed_config).train(data, checkpoint, resume=checkpoint)
    after = torch.load(checkpoint, map_location="cpu", weights_only=True)
    after_steps = [
        int(state["step"].item())
        for state in after["optimizer_state"]["state"].values()
        if "step" in state
    ]

    assert summary.start_step == 1
    assert summary.final_step == 2
    assert after_steps and set(after_steps) == {2}

def test_resume_appends_metrics_and_refreshes_checkpoint_fingerprint(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    metrics = tmp_path / "metrics.jsonl"
    write_algorithm_dataset(data, count=8)
    config = TrainingConfig(
        steps=1,
        batch_size=2,
        context=16,
        embedding_dim=8,
        ff_dim=16,
        heads=2,
        lr=1e-3,
        validation_split=0.0,
        seed=11,
        log_every=1,
        warmup_steps=1,
        min_lr_ratio=0.2,
    )

    TrainingPipeline(config).train(data, checkpoint, metrics_path=metrics)
    first = torch.load(checkpoint, map_location="cpu", weights_only=True)
    first_fingerprint = first["metrics_fingerprint"]
    first_rows = metrics.read_text(encoding="utf-8").splitlines()
    assert len(first_rows) == 1

    resumed_config = TrainingConfig(**{**config.__dict__, "steps": 1})
    summary = TrainingPipeline(resumed_config).train(
        data, checkpoint, resume=checkpoint, metrics_path=metrics
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines()]

    assert summary.start_step == 1
    assert summary.final_step == 2
    assert [row["step"] for row in rows] == [1, 2]
    assert payload["metrics_fingerprint"] != first_fingerprint
    assert payload["metrics_fingerprint"] == __import__("hashlib").sha256(
        metrics.read_bytes()
    ).hexdigest()
    assert payload["scheduler"]["optimizer_steps"] == 2
    assert payload["scheduler"]["total_steps"] == 2

