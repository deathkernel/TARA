"""Compatibility layer for TARA dataset loading.

The canonical dataset definitions live in src.dataset_registry. This module
keeps the earlier public loader names stable while routing all dataset access
through that single registry.
"""

from src.dataset_registry import DATASETS, get_dataset_spec, load_text_slice


def list_datasets():
    """Return registered dataset keys."""
    return sorted(DATASETS)


def describe_dataset(name):
    """Return a serializable description of one registered dataset."""
    spec = get_dataset_spec(name)
    config = "wikitext-2-raw-v1" if name.strip().lower() == "wikitext2" else None
    return {
        "dataset_id": spec.dataset_id,
        "config": config,
        "description": spec.role,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
    }


def load_dataset_text(name, max_chars=512, split="train"):
    """Load a small streamed text slice from a registered dataset."""
    return load_text_slice(name, max_chars=max_chars, split=split)


def load_tinystories_text(max_chars=512, split="train"):
    """Backward-compatible TinyStories loader."""
    return load_dataset_text("tinystories", max_chars=max_chars, split=split)
