"""Small, explicit dataset registry for TARA research experiments.

The goal is not to download giant corpora. Each dataset has a documented role
and can be sampled through the Hugging Face streaming interface so a normal PC
can run controlled experiments.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    train_split: str
    validation_split: str | None
    text_field: str
    role: str


DATASETS = {
    "tinystories": DatasetSpec(
        name="TinyStories",
        dataset_id="roneneldan/TinyStories",
        train_split="train",
        validation_split="validation",
        text_field="text",
        role="small language-model learning and generalization",
    ),
    "wikitext2": DatasetSpec(
        name="WikiText-2",
        dataset_id="Salesforce/wikitext",
        train_split="train",
        validation_split="validation",
        text_field="text",
        role="less synthetic language modeling benchmark",
    ),
}


def get_dataset_spec(name):
    """Return a registered dataset specification."""
    key = name.strip().lower()
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; available: {available}")
    return DATASETS[key]


def load_text_slice(name, max_chars=4096, split="train"):
    """Stream one deterministic text slice without loading the full corpus."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    spec = get_dataset_spec(name)
    if split not in {spec.train_split, spec.validation_split}:
        raise ValueError(f"unsupported split {split!r} for {spec.name}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "dataset loading requires the 'datasets' package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    dataset = load_dataset(spec.dataset_id, "wikitext-2-raw-v1" if name.lower() == "wikitext2" else None,
                           split=split, streaming=True)
    chunks = []
    total = 0
    for example in dataset:
        text = example.get(spec.text_field, "")
        if not text:
            continue
        remaining = max_chars - total
        chunks.append(text[:remaining])
        total += min(len(text), remaining)
        if total >= max_chars:
            break
    result = "\n".join(chunks)[:max_chars]
    if not result:
        raise ValueError(f"{spec.name} returned no text for split {split!r}")
    return result
