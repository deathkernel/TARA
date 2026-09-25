"""Compatibility wrapper; canonical streaming loader lives in src.tara_mind.data.streaming."""

from src.tara_mind.data.streaming import describe_dataset, list_datasets, load_dataset_text, load_tinystories_text

__all__ = ["describe_dataset", "list_datasets", "load_dataset_text", "load_tinystories_text"]
