"""Validate TARA training prerequisites without starting training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.dataset_audit import DatasetAuditor
from src.tokenizer import CharTokenizer
from src.training_pipeline import TrainingConfig, _tokenize_corpus, load_training_texts


def run_preflight(data: str | Path, *, context: int = 128, validation_split: float = 0.1) -> dict:
    path = Path(data)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    check("dataset_exists", path.is_file(), str(path))
    if not path.is_file():
        return {"ready": False, "checks": checks}

    try:
        texts = load_training_texts(path)
        check("dataset_load", True, f"records={len(texts)}")
    except Exception as exc:
        check("dataset_load", False, f"{type(exc).__name__}: {exc}")
        return {"ready": False, "checks": checks}

    audit = DatasetAuditor().audit(texts)
    check("dataset_audit", audit.healthy, f"records={audit.records} issues={len(audit.issues)} fingerprint={audit.fingerprint}")

    try:
        train_texts = texts
        tokenizer = CharTokenizer("\n".join(train_texts))
        tokens = _tokenize_corpus(train_texts, tokenizer, context)
        check("context_fit", True, f"tokens={len(tokens)} context={context} vocab={tokenizer.vocab_size}")
    except Exception as exc:
        check("context_fit", False, f"{type(exc).__name__}: {exc}")

    try:
        import torch
        check("torch_import", True, f"torch={torch.__version__}")
        check("device", True, "cuda" if torch.cuda.is_available() else "cpu")
    except Exception as exc:
        check("torch_import", False, f"{type(exc).__name__}: {exc}")

    try:
        TrainingConfig(context=context, validation_split=validation_split)
        check("training_config", True, "configuration accepted")
    except Exception as exc:
        check("training_config", False, f"{type(exc).__name__}: {exc}")

    ready = all(bool(item["passed"]) for item in checks)
    return {"ready": ready, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TARA training prerequisites without training")
    parser.add_argument("--data", default="data/algorithm_tasks.jsonl")
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--validation-split", type=float, default=0.1)
    args = parser.parse_args()
    report = run_preflight(args.data, context=args.context, validation_split=args.validation_split)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["ready"] else 1)


if __name__ == "__main__":
    main()
