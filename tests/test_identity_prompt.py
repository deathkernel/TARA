from src.brain import _tara_system_prompt


def test_tara_identity_prompt_names_project_and_configured_model():
    prompt = _tara_system_prompt("llama3.2:3b")

    assert "TARA — Tiny Artificial Reasoning Architecture" in prompt
    assert "Configured language model: llama3.2:3b." in prompt
    assert "served locally by Ollama" in prompt
    assert "Never claim to be an employee or member of Meta" in prompt
    assert "Do not invent who created TARA" in prompt
    assert "Saving memories" in prompt


def test_tara_identity_prompt_does_not_guess_an_unconfigured_model():
    prompt = _tara_system_prompt(None)

    assert "Configured language model: unknown (not explicitly configured)." in prompt
    assert "llama3.2:3b" not in prompt
