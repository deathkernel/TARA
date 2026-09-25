import math

import pytest

torch = pytest.importorskip("torch")

from src.brain import TARABrain
from src.language_model import TinyLanguageModel
from src.tokenizer import CharTokenizer
from src.torch_language_model import FastTinyLanguageModel


def make_brain():
    text = "tara can reason with context and memory."
    tokenizer = CharTokenizer(text)
    model = TinyLanguageModel(tokenizer.vocab_size, seed=5)
    return TARABrain(model, tokenizer, seed=5)


def test_brain_connects_observation_memory_and_generation():
    brain = make_brain()
    response = brain.respond("tara", remember_key="prompt", max_new_tokens=3)
    assert response.text
    assert "tara" in response.temporal_context
    assert response.observations[-1] == "tara"
    assert response.recalled


def test_brain_supports_goal_planning_and_verification():
    brain = make_brain()
    brain.set_goal("finish task", ("done",))
    brain.plan(["inspect", "finish"])
    assert brain.agent.next_task().task == "inspect"
    brain.verify_result("ok", "ok")
    assert brain.agent.next_task().task == "finish"


def test_brain_top_p_generation_is_finite_and_bounded():
    brain = make_brain()
    output = brain.generate("tara", max_new_tokens=5, top_p=0.8, temperature=1.0)
    assert output
    assert len(output) >= len("tara")
    assert all(math.isfinite(value) for value in brain.model.forward_numeric(brain.tokenizer.encode("tara"))[-1])


def test_brain_rejects_invalid_generation_arguments():
    brain = make_brain()
    for kwargs in ({"max_new_tokens": 0}, {"temperature": 0}, {"top_p": 0}, {"top_p": 1.1}):
        try:
            brain.generate("tara", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid generation argument should fail")


def test_brain_can_load_phase14_checkpoint(tmp_path):
    text = "tara learns from verified examples."
    tokenizer = CharTokenizer(text)
    model = FastTinyLanguageModel(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=8,
        ff_dim=16,
        num_heads=2,
        max_context=16,
        seed=3,
    )
    checkpoint = tmp_path / "brain.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": {
                "vocab_size": tokenizer.vocab_size,
                "embedding_dim": 8,
                "ff_dim": 16,
                "num_heads": 2,
                "max_context": 16,
            },
            "tokenizer": {"itos": tokenizer.itos, "stoi": tokenizer.stoi},
        },
        checkpoint,
    )
    brain = TARABrain.from_checkpoint(checkpoint, seed=3)
    assert brain.tokenizer.vocab_size == tokenizer.vocab_size
    assert brain.generate("tara", max_new_tokens=2)
