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
    return TrainingConfig(steps=steps, batch_size=2, context=8, embedding_dim=8, ff_dim=16, heads=2, lr=1e-3, validation_split=0.25, seed=11, log_every=2)


def test_phase14_writes_self_contained_checkpoint(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=8)
    summary = TrainingPipeline(tiny_config()).train(data, checkpoint)
    assert summary.start_step == 0
    assert summary.final_step == 2
    assert checkpoint.exists()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 5
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["hardening_config"]["gradient_accumulation_steps"] == 2
    assert payload["scheduler"]["optimizer_steps"] == 3
    assert payload["metrics_fingerprint"]


def test_bpe_tokenizer_is_default_and_persisted(tmp_path):
    data = tmp_path / "data.jsonl"
    checkpoint = tmp_path / "model.pt"
    write_algorithm_dataset(data, count=12)
    config = TrainingConfig(steps=1, batch_size=2, context=8, embedding_dim=8, ff_dim=16, heads=2, lr=1e-3, validation_split=0.25, seed=11, log_every=1, vocab_size=64)
    summary = TrainingPipeline(config).train(data, checkpoint)
    assert summary.final_step == 1
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["format_version"] == 5
    assert payload["tokenizer"]["type"] == "bpe"
    assert payload["tokenizer"]["merges"]

    
def test_tokenize_corpus_keeps_records_separate():
    from src.tokenizer import CharTokenizer
    from src.training_pipeline import _tokenize_corpus

    tokenizer = CharTokenizer("a" * 80 + "b" * 80)
    sequences = _tokenize_corpus(["a" * 80, "b" * 80], tokenizer, context=16)
    assert len(sequences) == 2
    assert sequences[0].shape[0] == 80
    assert sequences[1].shape[0] == 80


def test_sample_batch_never_crosses_record_boundary():
    from src.tokenizer import CharTokenizer
    from src.training_pipeline import _sample_batch, _tokenize_corpus

    tokenizer = CharTokenizer("a" * 80 + "b" * 80)
    sequences = _tokenize_corpus(["a" * 80, "b" * 80], tokenizer, context=16)
    x, y = _sample_batch(sequences, batch_size=32, context=16, device=torch.device("cpu"))
    assert x.shape == (32, 16)
    assert y.shape == (32, 16)
    for row in x:
        assert len(set(row.tolist())) == 1


def test_curriculum_v2_has_shorter_lesson_records():
    from pathlib import Path

    path = Path("data/curriculum_v2.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 24
    assert all(row["level"] in range(1, 13) for row in rows)
    assert all(row["text"].count("<|user|>") == row["text"].count("<|assistant|>") for row in rows)
    assert all(row["text"].count("<|user|>") >= 4 for row in rows)
