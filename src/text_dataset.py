"""Small streaming text-dataset utilities for TARA.

The dataset layer separates corpus choice from model/training code. TARA can
therefore compare datasets without rewriting the language-model pipeline.

Current built-in corpora:
- TinyStories
- WikiText-2 (raw)

Only small streamed slices are loaded for normal-PC experiments.
"""

DATASETS = {
    "tinystories": {
        "dataset_id": "roneneldan/TinyStories",
        "config": None,
        "description": "Synthetic/simple stories designed for small language models.",
    },
    "wikitext2": {
        "dataset_id": "Salesforce/wikitext",
        "config": "wikitext-2-raw-v1",
        "description": "Natural-language Wikipedia-derived benchmark text.",
    },
}


def list_datasets():
    """Return names of datasets supported by the TARA text loader."""
    return sorted(DATASETS)


def describe_dataset(name):
    """Return a copy of one dataset specification."""
    key = name.lower()
    if key not in DATASETS:
        raise KeyError(f"unknown dataset: {name}")
    return dict(DATASETS[key])


def load_dataset_text(name, max_chars=512, split="train"):
    """Stream a deterministic text slice from a registered dataset."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    key = name.lower()
    if key not in DATASETS:
        raise ValueError(
            f"unknown dataset: {name}. Choose from {list_datasets()}"
        )

    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be 'train', 'validation', or 'test'")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Dataset loading requires the 'datasets' package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    spec = DATASETS[key]
    kwargs = {"split": split, "streaming": True}

    if spec["config"] is None:
        dataset = load_dataset(spec["dataset_id"], **kwargs)
    else:
        dataset = load_dataset(spec["dataset_id"], spec["config"], **kwargs)

    chunks = []
    total = 0

    for example in dataset:
        text = example.get("text", "")
        if not text:
            continue

        remaining = max_chars - total
        chunks.append(text[:remaining])
        total += min(len(text), remaining)

        if total >= max_chars:
            break

    text = "\n".join(chunks)

    if not text:
        raise ValueError(f"{name}/{split} returned no usable text")

    return text[:max_chars]


def load_tinystories_text(max_chars=512, split="train"):
    """Backward-compatible TinyStories loader."""
    return load_dataset_text("tinystories", max_chars=max_chars, split=split)
