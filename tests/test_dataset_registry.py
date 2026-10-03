import pytest

from src.dataset_registry import _example_to_text, get_dataset_spec


def test_registered_datasets_have_expected_roles():
    tiny = get_dataset_spec("TinyStories")
    wiki = get_dataset_spec("wikitext2")
    assert tiny.dataset_id == "roneneldan/TinyStories"
    assert wiki.dataset_id == "Salesforce/wikitext"
    assert tiny.validation_split == "validation"
    assert wiki.text_field == "text"


def test_curriculum_uses_current_compatible_dataset_sources():
    assert get_dataset_spec("empathetic_dialogues").dataset_id == "lighteval/empathetic_dialogues"
    assert get_dataset_spec("blended_skill_talk").dataset_id == "TutorialGuide/blended-skill-talk-fixed"
    assert get_dataset_spec("daily_dialog").dataset_id == "DeepPavlov/daily_dialog"


def test_empathetic_parquet_schema_is_converted_to_training_text():
    spec = get_dataset_spec("empathetic_dialogues")
    text = _example_to_text(
        spec,
        {
            "input": "I am preparing for a marathon.",
            "references": ["You should train consistently.", "That sounds exciting."],
            "subsplit": "confident",
        },
    )
    assert "I am preparing for a marathon." in text
    assert "You should train consistently." in text
    assert "That sounds exciting." in text


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        get_dataset_spec("does-not-exist")
