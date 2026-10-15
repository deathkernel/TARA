"""Batch F release-integration tests for TARA's composed runtime.

These tests are local and deterministic: no Ollama server, remote API, generated
teacher data, checkpoint training, or checkpoint promotion is required.
"""
from src.brain import TARABrain, _tara_system_prompt
from src.experience_learning import Experience
from src.llm_provider_factory import provider_from_env
from src.memory import LongTermMemory
from src.tara_core import CoreConfig, CoreStatus, TARACore
from src.unified_cognitive_loop import ActionOutcome


class DummyModel:
    """Marker model; the integration test exercises orchestration, not inference."""


def test_phase_f_core_memory_learning_lifecycle_is_composed(tmp_path, monkeypatch):
    monkeypatch.delenv("TARA_LLM_BACKEND", raising=False)
    memory_path = tmp_path / "memory.json"
    memory = LongTermMemory(memory_path)
    memory.remember("preferred_language", "Hinglish")

    brain = TARABrain(DummyModel(), memory=memory, seed=17)
    core = TARACore(brain, config=CoreConfig(seed=17, max_cycles=4, max_events=3))
    assert core.health().healthy
    assert provider_from_env() is None

    report = core.run(
        "verify integrated runtime",
        ["inspect inputs", "record outcome"],
        ["input is available", "outcome is recorded"],
        action_executor=lambda request: ActionOutcome(
            request.step.description, True, score=1.0
        ),
        expected=lambda value: isinstance(value, str),
    )
    assert report.completed
    assert core.status is CoreStatus.READY

    learning = core.learn([
        Experience("verified-1", "input", "action", "success", 1.0, True),
        Experience("unverified-1", "input", "action", "unknown", 0.0, False),
    ])
    assert learning.updates

    memory.save()
    restored = LongTermMemory(memory_path)
    assert restored.recall("preferred_language") == "Hinglish"

    manifest_path = tmp_path / "runtime-manifest.json"
    snapshot = core.save_runtime_manifest(manifest_path)
    assert snapshot.fingerprint
    assert manifest_path.is_file()
    assert core.events()
    core.stop()
    assert core.status is CoreStatus.STOPPED


def test_phase_f_provider_selection_is_explicit_and_offline(monkeypatch):
    monkeypatch.setenv("TARA_LLM_BACKEND", "ollama")
    monkeypatch.setenv("TARA_LLAMA_MODEL", "llama3.2:1b")
    provider = provider_from_env()
    assert provider.model == "llama3.2:1b"
    # Constructing the provider must not make an inference request or start training.


def test_phase_f_identity_does_not_confuse_runtime_with_model():
    prompt = _tara_system_prompt("llama3.2:1b")
    assert "TARA — Tiny Artificial Reasoning Architecture" in prompt
    assert "Configured language model: llama3.2:1b." in prompt
    assert "Do not claim self-training, self-learning, or model-weight updates occur during chat." in prompt
