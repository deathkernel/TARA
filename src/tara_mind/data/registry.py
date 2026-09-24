"""Explicit dataset registry for TARA research experiments.

The active vNext conversational curriculum starts with SODA because it provides
million-scale social dialogue, broad interaction coverage, and a large
emotion-grounded subset. Datasets that are not active can remain registered
for reproducibility, but the vNext runner defaults to SODA.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    dataset_id: str
    train_split: str
    validation_split: str | None
    text_field: str | None
    role: str
    dialogue_field: str | None = None
    speaker_field: str | None = None


DATASETS = {
    "soda": DatasetSpec(
        name="SODA",
        dataset_id="allenai/soda",
        train_split="train",
        validation_split="validation",
        text_field=None,
        role=(
            "large-scale social dialogue for natural conversation, "
            "commonsense, interpersonal interaction, and emotion-aware behavior"
        ),
        dialogue_field="dialogue",
        speaker_field="speakers",
    ),
    "tinystories": DatasetSpec(
        name="TinyStories",
        dataset_id="roneneldan/TinyStories",
        train_split="train",
        validation_split="validation",
        text_field="text",
        role="legacy small-language-model warm-up corpus",
    ),
    "wikitext2": DatasetSpec(
        name="WikiText-2",
        dataset_id="Salesforce/wikitext",
        train_split="train",
        validation_split="validation",
        text_field="text",
        role="less synthetic language modeling benchmark",
    ),
    "fineweb_edu": DatasetSpec(
        name="FineWeb-Edu",
        dataset_id="HuggingFaceFW/fineweb-edu",
        train_split="train",
        validation_split=None,
        text_field="text",
        role="educational web text for general language and knowledge learning",
    ),
    "gsm8k": DatasetSpec(
        name="GSM8K",
        dataset_id="openai/gsm8k",
        train_split="train",
        validation_split="test",
        text_field="question",
        role="grade-school mathematical reasoning",
    ),
    "fineweb": DatasetSpec(
        name="FineWeb",
        dataset_id="HuggingFaceFW/fineweb",
        train_split="train",
        validation_split=None,
        text_field="text",
        role="broad large-scale English web pretraining data",
    ),
    "the_stack_v2_smol": DatasetSpec(
        name="The Stack v2 Smol",
        dataset_id="bigcode/the-stack-v2-train-smol",
        train_split="train",
        validation_split=None,
        text_field="content",
        role="legacy programming-data registry entry",
    ),
}


def list_datasets():
    """Return registered dataset keys."""
    return sorted(DATASETS)


def get_dataset_spec(name):
    """Return a registered dataset specification."""
    if not isinstance(name, str):
        raise TypeError("dataset name must be a string")
    key = name.strip().lower()
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; available: {available}")
    return DATASETS[key]


def describe_dataset(name):
    """Return a serializable description for compatibility with older code."""
    spec = get_dataset_spec(name)
    config = None
    if name.strip().lower() == "wikitext2":
        config = "wikitext-2-raw-v1"
    elif name.strip().lower() == "gsm8k":
        config = "main"
    return {
        "dataset_id": spec.dataset_id,
        "config": config,
        "description": spec.role,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "text_field": spec.text_field,
        "dialogue_field": spec.dialogue_field,
        "speaker_field": spec.speaker_field,
    }


def _example_to_text(spec, example):
    if spec.dialogue_field:
        dialogue = example.get(spec.dialogue_field)
        if not isinstance(dialogue, list):
            return ""
        speakers = example.get(spec.speaker_field, []) if spec.speaker_field else []
        lines = []
        for index, utterance in enumerate(dialogue):
            if not isinstance(utterance, str) or not utterance.strip():
                continue
            speaker = "Speaker"
            if isinstance(speakers, list) and index < len(speakers):
                candidate = speakers[index]
                if isinstance(candidate, str) and candidate.strip():
                    speaker = candidate.strip()
            lines.append(f"{speaker}: {utterance.strip()}")
        return "\n".join(lines)
    if not spec.text_field:
        return ""
    value = example.get(spec.text_field, "")
    return value if isinstance(value, str) else ""


def load_text_slice(name, max_chars=4096, split="train", seed=42, shuffle=True):
    """Stream a bounded, reproducible text slice without loading the full corpus."""
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise TypeError("max_chars must be an integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    spec = get_dataset_spec(name)
    allowed_splits = {spec.train_split}
    if spec.validation_split is not None:
        allowed_splits.add(spec.validation_split)
    if split not in allowed_splits:
        raise ValueError(f"unsupported split {split!r} for {spec.name}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "dataset loading requires the 'datasets' package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    config = None
    if name.strip().lower() == "wikitext2":
        config = "wikitext-2-raw-v1"
    elif name.strip().lower() == "gsm8k":
        config = "main"

    kwargs = {"split": split, "streaming": True}
    if config is None:
        dataset = load_dataset(spec.dataset_id, **kwargs)
    else:
        dataset = load_dataset(spec.dataset_id, config, **kwargs)

    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=10_000)

    chunks = []
    total = 0

    for example in dataset:
        text = _example_to_text(spec, example)
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
