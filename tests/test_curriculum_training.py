from scripts.train.train_tara_curriculum import (
    DEFAULT_ORDER,
    _load_generated_stage,
    parse_order,
)


def test_curriculum_includes_method_and_automation_stages():
    assert DEFAULT_ORDER == (
        "A_social",
        "B_math",
        "C_physics",
        "D_science",
        "E_method",
        "F_automation",
    )


def test_generated_scientific_method_stage_is_deterministic():
    first = _load_generated_stage("E_method", 2000)
    second = _load_generated_stage("E_method", 2000)
    assert first == second
    assert "Observation:" in first[0]
    assert "Hypothesis:" in first[0]
    assert "Uncertainty:" in first[0]


def test_generated_tool_stage_contains_verification_boundary():
    train, validation = _load_generated_stage("F_automation", 2000)
    assert train == _load_generated_stage("F_automation", 2000)[0]
    assert "Tool policy:" in train
    assert "verify" in train.lower()
    assert validation


def test_parse_order_accepts_full_default_curriculum():
    assert parse_order(",".join(DEFAULT_ORDER)) == list(DEFAULT_ORDER)
