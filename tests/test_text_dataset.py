from src.dataset_registry import DATASETS, describe_dataset, list_datasets


def test_dataset_registry_contains_core_corpora():
    assert {"tinystories", "wikitext2"} <= set(list_datasets())


def test_dataset_description_is_copy():
    description = describe_dataset("tinystories")
    assert description["dataset_id"] == DATASETS["tinystories"].dataset_id
    description["description"] = "changed"
    assert DATASETS["tinystories"].role != "changed"


def test_unknown_dataset_rejected():
    try:
        describe_dataset("does-not-exist")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown datasets must raise KeyError")
