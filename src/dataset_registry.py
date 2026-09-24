"""Compatibility wrapper; canonical registry lives in src.tara_mind.data.registry."""

from src.tara_mind.data.registry import DATASETS, DatasetSpec, describe_dataset, get_dataset_spec, list_datasets, load_text_slice

__all__ = ["DATASETS", "DatasetSpec", "describe_dataset", "get_dataset_spec", "list_datasets", "load_text_slice"]
