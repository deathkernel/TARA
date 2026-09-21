"""Small text-dataset utilities for TARA language-model experiments.

The first real corpus is TinyStories. The original TinyStories work studies
small language models trained on short, simple stories, including models with
one Transformer block. TARA uses only a tiny streamed slice so the experiment
remains feasible on a normal PC.
"""


def load_tinystories_text(max_chars=512, split="train"):
    """Stream TinyStories and return a deterministic small text slice.

    Only enough examples to reach ``max_chars`` are consumed. The full
    TinyStories corpus is intentionally not downloaded into memory.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if split not in {"train", "validation"}:
        raise ValueError("split must be 'train' or 'validation'")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "TinyStories loading requires the 'datasets' package. "
            "Install it with: python -m pip install datasets"
        ) from exc

    dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
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
        raise ValueError("TinyStories returned no text")
    return text[:max_chars]
