"""Explicit dataset registry for TARA research experiments.

The active curriculum avoids programming data. Each dataset has a deterministic
text adapter so reasoning examples become training sequences without loading an
entire remote corpus into memory.
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
    config: str | None = None
    adapter: str = "generic"


DATASETS = {
    "soda": DatasetSpec("SODA", "allenai/soda", "train", "validation", None,
                        "social dialogue, commonsense and interpersonal interaction",
                        "dialogue", "speakers", adapter="dialogue"),
    "empathetic_dialogues": DatasetSpec(
        "EmpatheticDialogues", "facebook/empathetic_dialogues", "train", "validation",
        None, "emotion-grounded open-domain conversation", adapter="empathetic"
    ),
    "daily_dialog": DatasetSpec(
        "DailyDialog", "li2017dailydialog/daily_dialog", "train", "validation",
        None, "human-written everyday dialogue with emotion and intent labels",
        dialogue_field="dialog", adapter="dialogue_list"
    ),
    "blended_skill_talk": DatasetSpec(
        "BlendedSkillTalk", "anezatra/blended-skill-talk", "train", "validation",
        None, "persona, empathy, knowledge and dialogue flow",
        adapter="blended"
    ),
    "gsm8k": DatasetSpec(
        "GSM8K", "openai/gsm8k", "train", "test", None,
        "grade-school mathematical reasoning", config="main", adapter="gsm8k"
    ),
    "competition_math": DatasetSpec(
        "MATH", "jeggers/competition_math", "training", "test", None,
        "competition mathematics with worked solutions", adapter="math"
    ),
    "physics_eval": DatasetSpec(
        "PhysicsEval", "IUTVanguard/PhysicsEval", "train", "test", None,
        "annotated physics problems with stepwise solutions", adapter="physics"
    ),
    "sciq": DatasetSpec(
        "SciQ", "allenai/sciq", "train", "validation", None,
        "science questions with supporting evidence", adapter="sciq"
    ),
    "ai2_arc": DatasetSpec(
        "AI2 ARC", "allenai/ai2_arc", "train", "validation", None,
        "grade-school science reasoning", config="ARC-Challenge", adapter="arc"
    ),
    "tinystories": DatasetSpec(
        "TinyStories", "roneneldan/TinyStories", "train", "validation", "text",
        "legacy small-language-model warm-up corpus"
    ),
    "wikitext2": DatasetSpec(
        "WikiText-2", "Salesforce/wikitext", "train", "validation", "text",
        "language-modeling benchmark", config="wikitext-2-raw-v1"
    ),
    "fineweb_edu": DatasetSpec(
        "FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "train", None, "text",
        "educational web text for general language and knowledge learning"
    ),
}


def list_datasets():
    return sorted(DATASETS)


def get_dataset_spec(name):
    if not isinstance(name, str):
        raise TypeError("dataset name must be a string")
    key = name.strip().lower()
    if key not in DATASETS:
        available = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {name!r}; available: {available}")
    return DATASETS[key]


def describe_dataset(name):
    spec = get_dataset_spec(name)
    return {
        "dataset_id": spec.dataset_id,
        "config": spec.config,
        "description": spec.role,
        "train_split": spec.train_split,
        "validation_split": spec.validation_split,
        "text_field": spec.text_field,
        "dialogue_field": spec.dialogue_field,
        "speaker_field": spec.speaker_field,
        "adapter": spec.adapter,
    }


def _clean(value):
    return str(value).strip() if isinstance(value, str) and value.strip() else ""


def _example_to_text(spec, example):
    adapter = spec.adapter

    if adapter == "dialogue":
        dialogue = example.get(spec.dialogue_field or "")
        speakers = example.get(spec.speaker_field or "", [])
        if not isinstance(dialogue, list):
            return ""
        lines = []
        for i, utterance in enumerate(dialogue):
            utterance = _clean(utterance)
            if not utterance:
                continue
            speaker = _clean(speakers[i]) if isinstance(speakers, list) and i < len(speakers) else "Speaker"
            lines.append(f"{speaker or 'Speaker'}: {utterance}")
        return "\n".join(lines)

    if adapter == "dialogue_list":
        dialogue = example.get(spec.dialogue_field or "")
        if not isinstance(dialogue, list):
            return ""
        return "\n".join(f"Speaker: {_clean(x)}" for x in dialogue if _clean(x))

    if adapter == "empathetic":
        fields = [_clean(example.get("context")), _clean(example.get("prompt")), _clean(example.get("utterance"))]
        return "\n".join(x for x in fields if x)

    if adapter == "blended":
        fields = []
        for key in ("additional_context", "context", "previous_utterance", "free_messages", "guided_messages"):
            value = example.get(key)
            if isinstance(value, list):
                fields.extend(_clean(x) for x in value if _clean(x))
            else:
                value = _clean(value)
                if value:
                    fields.append(value)
        return "\n".join(fields)

    if adapter == "gsm8k":
        question = _clean(example.get("question"))
        answer = _clean(example.get("answer"))
        return f"Question: {question}\nAnswer: {answer}".strip()

    if adapter == "math":
        fields = [_clean(example.get("problem")), _clean(example.get("solution")), _clean(example.get("level")), _clean(example.get("type"))]
        return "\n".join(x for x in fields if x)

    if adapter == "physics":
        fields = [
            _clean(example.get("problem")),
            _clean(example.get("simplified_problem_statement")),
            _clean(example.get("elaborated_solution_steps")),
            _clean(example.get("final_answers_in_brief")),
        ]
        return "\n".join(x for x in fields if x)

    if adapter == "sciq":
        fields = [_clean(example.get("question"))]
        for key in ("distractor1", "distractor2", "distractor3", "correct_answer", "support"):
            value = _clean(example.get(key))
            if value:
                fields.append(value)
        return "\n".join(fields)

    if adapter == "arc":
        question = _clean(example.get("question"))
        choices = example.get("choices", {})
        labels = choices.get("label", []) if isinstance(choices, dict) else []
        texts = choices.get("text", []) if isinstance(choices, dict) else []
        options = []
        if isinstance(labels, list) and isinstance(texts, list):
            for label, text in zip(labels, texts):
                options.append(f"{_clean(label)}. {_clean(text)}")
        answer = _clean(example.get("answerKey"))
        return "\n".join(x for x in (
            f"Question: {question}" if question else "",
            "Choices: " + " ".join(options) if options else "",
            f"Answer: {answer}" if answer else "",
        ) if x)

    if not spec.text_field:
        return ""
    return _clean(example.get(spec.text_field))


def load_text_slice(name, max_chars=4096, split="train", seed=42, shuffle=True):
    if not isinstance(max_chars, int) or isinstance(max_chars, bool):
        raise TypeError("max_chars must be an integer")
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    spec = get_dataset_spec(name)
    allowed = {spec.train_split}
    if spec.validation_split is not None:
        allowed.add(spec.validation_split)
    if split not in allowed:
        raise ValueError(f"unsupported split {split!r} for {spec.name}")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("install 'datasets' with: python -m pip install datasets") from exc

    kwargs = {"split": split, "streaming": True}
    if spec.config is None:
        dataset = load_dataset(spec.dataset_id, **kwargs)
    else:
        dataset = load_dataset(spec.dataset_id, spec.config, **kwargs)
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


def load_dataset_text(name, max_chars=4096, split="train", seed=42, shuffle=True):
    return load_text_slice(name, max_chars, split, seed, shuffle)


def load_tinystories_text(max_chars=512, split="train"):
    return load_dataset_text("tinystories", max_chars=max_chars, split=split)
