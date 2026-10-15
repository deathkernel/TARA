"""Release integration checks for the non-persistent TARA runtime."""
import pytest

from src.brain import TARABrain, _tara_system_prompt
from src.experience_learning import Experience
from src.llm_provider_factory import provider_from_env
from src.memory import LongTermMemory
from src.tara_core import CoreConfig, CoreStatus, TARACore
from src.unified_cognitive_loop import ActionOutcome


class DummyModel:
    pass


def test_runtime_composes_without_persistent_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("TARA_LLM_BACKEND", raising=False)
    memory = LongTermMemory(tmp_path / "memory.json")
    memory.remember("preferred_language", "Hinglish")
    assert memory.recall("preferred_language") is None
    assert memory.retrieve("preferred_language") == []
    with pytest.raises(RuntimeError, match="removed"):
        memory.save()

    brain = TARABrain(DummyModel(), memory=memory, seed=17)
    core = TARACore(brain, config=CoreConfig(seed=17, max_cycles=4, max_events=3))
    assert core.health().healthy
    assert provider_from_env() is None
    report = core.run(
        "verify integrated runtime",
        ["inspect inputs", "record outcome"],
        ["input is available", "outcome is recorded"],
        action_executor=lambda request: ActionOutcome(request.step.description, True, score=1.0),
        expected=lambda value: isinstance(value, str),
    )
    assert report.completed
    assert core.status is CoreStatus.READY
    learning = core.learn([Experience("verified-1", "input", "action", "success", 1.0, True)])
    assert learning.updates
    manifest_path = tmp_path / "runtime-manifest.json"
    snapshot = core.save_runtime_manifest(manifest_path)
    assert snapshot.fingerprint
    assert manifest_path.is_file()
    assert core.events()
    core.stop()
    assert core.status is CoreStatus.STOPPED


def test_provider_selection_is_explicit_and_offline(monkeypatch):
    monkeypatch.setenv("TARA_LLM_BACKEND", "ollama")
    monkeypatch.setenv("TARA_LLAMA_MODEL", "llama3.2:1b")
    provider = provider_from_env()
    assert provider.model == "llama3.2:1b"


def test_identity_prompt_does_not_claim_training_during_chat():
    prompt = _tara_system_prompt("llama3.2:1b")
    assert "TARA — Tiny Artificial Reasoning Architecture" in prompt
    assert "Configured language model: llama3.2:1b." in prompt
    assert "Do not claim self-training, self-learning, or model-weight updates occur during chat." in prompt
    assert "Conversation memory is not persisted" in prompt
