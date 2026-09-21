import pytest

from src.perception import DocumentState, SystemState, perceive_document, perceive_system_state, summarize_screen_state


def test_document_perception_reads_bounded_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("TARA learns", encoding="utf-8")
    state = perceive_document(path)
    assert isinstance(state, DocumentState)
    assert state.text == "TARA learns"
    assert state.suffix == ".txt"
    assert state.size_bytes == len("TARA learns".encode("utf-8"))


def test_document_perception_rejects_missing_and_oversized_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        perceive_document(tmp_path / "missing.txt")
    path = tmp_path / "large.txt"
    path.write_text("abcdef", encoding="utf-8")
    with pytest.raises(ValueError):
        perceive_document(path, max_bytes=5)


def test_system_state_is_deterministic_and_normalized():
    state = perceive_system_state({"memory": 3, "mode": "idle"})
    assert isinstance(state, SystemState)
    assert state.facts == (("memory", "3"), ("mode", "idle"))
    assert state.as_dict() == {"memory": "3", "mode": "idle"}


def test_screen_summary_is_deterministic():
    elements = [
        {"type": "button", "label": "Run"},
        {"type": "text", "text": "Ready"},
    ]
    assert summarize_screen_state(elements) == "button: Run\ntext: Ready"


def test_perception_validates_inputs():
    with pytest.raises(ValueError):
        perceive_document("anything", max_bytes=0)
    with pytest.raises(TypeError):
        perceive_system_state([])
    with pytest.raises(TypeError):
        summarize_screen_state(["button"])
