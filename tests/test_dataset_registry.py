import pytest

from src.dataset_registry import get_dataset_spec


def test_registered_datasets_have_expected_roles():
    tiny = get_dataset_spec("TinyStories")
    wiki = get_dataset_spec("wikitext2")
    assert tiny.dataset_id == "roneneldan/TinyStories"
    assert wiki.dataset_id == "Salesforce/wikitext"
    assert tiny.validation_split == "validation"
    assert wiki.text_field == "text"


def test_unknown_dataset_is_rejected():
    with pytest.raises(KeyError):
        get_dataset_spec("does-not-exist")
