"""Canonical dataset registry delegated to the single root implementation.

Keeping one registry prevents curriculum code from using stale or divergent
dataset adapters.
"""

from src.dataset_registry import (
    DATASETS,
    DatasetSpec,
    describe_dataset,
    get_dataset_spec,
    list_datasets,
    load_dataset_text,
    load_text_slice,
)

__all__ = [
    "DATASETS",
    "DatasetSpec",
    "describe_dataset",
    "get_dataset_spec",
    "list_datasets",
    "load_dataset_text",
    "load_text_slice",
]
