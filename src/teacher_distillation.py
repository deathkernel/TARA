"""Generate candidate multi-domain training examples from a local LLM teacher.

This writes data only. It does not train or promote a TARA checkpoint.
Teacher outputs remain unverified candidates until independently evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Protocol


class TextProvider(Protocol):
    def complete(self, prompt: str, *, system: str | None = None,
                 max_completion_tokens: int = 1024, temperature: float = 0.6) -> str: ...


DOMAIN_TASKS: dict[str, tuple[str, ...]] = {
    "math": (
        "Solve and show a checkable derivation: {n} + {m} * 3.",
        "Explain the formula for percentage increase from {n} to {m}.",
        "Create and solve a two-step word problem involving {n} and {m}.",
        "Explain why division by zero is undefined, with an example.",
        "Give a method for checking whether an arithmetic answer is plausible.",
    ),
    "reasoning": (
        "Use only these facts: all dax are wugs; {n} is a dax. What follows logically?",
        "A task takes {n} minutes and its prerequisite takes {m} minutes. Explain total sequential time.",
        "Give a small logic puzzle, solve it, and separate assumptions from conclusions.",
        "Explain correlation versus causation with a concrete example.",
        "Give a counterexample to an overgeneralized claim and explain why it disproves it.",
    ),
    "coding": (
        "Write a Python function returning the largest integer in a list; handle edge cases and test it.",
        "Write a Python function removing duplicates while preserving order. Explain complexity and test it.",
        "Debug this Python snippet conceptually: for i in range(3): print(items[i]). What could go wrong?",
        "Write a robust Python function parsing a positive integer from text.",
        "Explain how to unit-test sorting records by a named field.",
    ),
    "algorithms": (
        "Explain binary search, preconditions, and complexity with an example.",
        "Compare a list and hash map for lookup; discuss average and worst-case complexity.",
        "Design duplicate detection and explain time/space trade-offs.",
        "Explain breadth-first search and when it finds a shortest path.",
        "Give pseudocode for merging sorted lists and analyze complexity.",
    ),
    "language": (
        "Explain a technical idea in simple Hinglish, then give an English summary.",
        "Rewrite a vague request as three precise requirements and explain the ambiguity.",
        "Summarize a hypothetical project update in three bullets without inventing facts.",
        "Give a respectful correction when a user states something incorrect.",
        "Explain how to ask one focused clarification question for an ambiguous requirement.",
    ),
    "knowledge_science": (
        "Explain the scientific method and distinguish a hypothesis from evidence.",
        "Explain weather versus climate without overstating certainty.",
        "Describe controlled experiments and common confounders.",
        "Explain what a software benchmark can and cannot prove about a model.",
        "Explain at a high level how batteries store and release energy.",
    ),
    "planning": (
        "Plan a project with {n} tasks, dependencies, milestones, and completion checks.",
        "Create a debugging plan from reproduction to regression tests.",
        "Prioritize tasks when impact, urgency, and effort differ.",
        "Plan a safe software update with pre-checks, backup, validation, and rollback.",
        "Break an unknown blocker into verifiable next steps.",
    ),
    "tool_use_verification": (
        "A command claims a file was created. Explain how to verify existence and expected content.",
        "Explain why an agent must not claim it ran tests without actual test output.",
        "Propose a safe workflow for editing a file, inspecting the diff, testing, and reporting.",
        "A tool errors halfway through a task. Explain recovery without duplicating side effects.",
        "Explain when an agent should ask before destructive or externally visible actions.",
    ),
    "safety_and_uncertainty": (
        "Explain how to respond when evidence is insufficient without inventing details.",
        "Describe a privacy-conscious approach to storing user preferences locally.",
        "Explain why generated training examples should be checked for errors and duplicates.",
        "Give a safe response to a request that could delete files, emphasizing confirmation and backups.",
        "Explain how to distinguish verified facts from assumptions.",
    ),
}


def build_prompt(domain: str, task: str, index: int) -> str:
    n, m = 7 + (index * 3 % 19), 2 + (index * 5 % 11)
    question = task.format(n=n, m=m)
    if index >= len(DOMAIN_TASKS[domain]):
        question += f" Use a fresh variant number {index + 1}; avoid repeating wording."
    return question


def generate_teacher_dataset(
    provider: TextProvider,
    output: str | Path,
    *,
    per_domain: int = 10,
    max_completion_tokens: int = 512,
    temperature: float = 0.2,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write teacher responses as JSONL records with provenance and fingerprints."""
    if per_domain <= 0:
        raise ValueError("per_domain must be positive")
    if max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    if not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0 and 2")

    destination = Path(output)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; choose another path or pass --overwrite")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    counts = {domain: 0 for domain in DOMAIN_TASKS}
    seen: set[str] = set()
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            for domain, tasks in DOMAIN_TASKS.items():
                for index in range(per_domain):
                    question = build_prompt(domain, tasks[index % len(tasks)], index)
                    answer = provider.complete(
                        question,
                        system=(
                            "You are a careful teacher generating a candidate example for a small "
                            "language model. Be accurate, self-contained, concise, and explicit "
                            "about assumptions. Include tests for code where useful. Never claim "
                            "external actions were performed."
                        ),
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                    ).strip()
                    if len(answer) < 12:
                        continue
                    fingerprint = hashlib.sha256(
                        (domain + "\n" + question + "\n" + answer).encode("utf-8")
                    ).hexdigest()
                    if fingerprint in seen:
                        continue
                    seen.add(fingerprint)
                    record = {
                        "text": f"Domain: {domain}\nInstruction: {question}\nAnswer: {answer}",
                        "domain": domain,
                        "instruction": question,
                        "answer": answer,
                        "source": "ollama_teacher_generated",
                        "teacher_model": getattr(provider, "model", "injected-provider"),
                        "fingerprint": fingerprint,
                        "review_status": "unverified_candidate",
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    counts[domain] += 1
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    counts["total"] = sum(counts.values())
    return counts



def audit_teacher_dataset(path: str | Path) -> dict[str, object]:
    """Audit candidate JSONL structure and provenance; never certify semantic correctness."""
    source_path = Path(path)
    issues: list[dict[str, object]] = []
    seen_fingerprints: set[str] = set()
    duplicate_fingerprints = 0
    records = 0
    valid_records = 0

    if not source_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    with source_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                issues.append({"line": line_number, "errors": ["blank line is not valid JSONL data"]})
                continue
            records += 1
            errors: list[str] = []
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                issues.append({"line": line_number, "errors": [f"invalid JSON: {exc.msg}"]})
                continue
            if not isinstance(row, dict):
                issues.append({"line": line_number, "errors": ["record must be a JSON object"]})
                continue

            domain = row.get("domain")
            instruction = row.get("instruction")
            answer = row.get("answer")
            fingerprint = row.get("fingerprint")
            if domain not in DOMAIN_TASKS:
                errors.append("unknown or missing domain")
            if not isinstance(instruction, str) or not instruction.strip():
                errors.append("instruction must be a non-empty string")
            if not isinstance(answer, str) or len(answer.strip()) < 12:
                errors.append("answer must contain at least 12 non-whitespace characters")
            if row.get("source") != "ollama_teacher_generated":
                errors.append("unexpected or missing source provenance")
            if not isinstance(row.get("teacher_model"), str) or not row["teacher_model"].strip():
                errors.append("teacher_model must be a non-empty string")
            if row.get("review_status") != "unverified_candidate":
                errors.append("review_status must remain unverified_candidate")
            if isinstance(domain, str) and isinstance(instruction, str) and isinstance(answer, str):
                expected_text = f"Domain: {domain}\\nInstruction: {instruction}\\nAnswer: {answer}"
                if row.get("text") != expected_text:
                    errors.append("text field does not match domain/instruction/answer")
                expected_fingerprint = hashlib.sha256(
                    f"{domain}\\n{instruction}\\n{answer}".encode("utf-8")
                ).hexdigest()
                if fingerprint != expected_fingerprint:
                    errors.append("fingerprint mismatch")
            if not isinstance(fingerprint, str) or len(fingerprint) != 64:
                errors.append("fingerprint must be a SHA-256 hex digest")
            elif fingerprint in seen_fingerprints:
                duplicate_fingerprints += 1
                errors.append("duplicate fingerprint")
            else:
                seen_fingerprints.add(fingerprint)

            if errors:
                issues.append({"line": line_number, "errors": errors})
            else:
                valid_records += 1

    return {
        "path": str(source_path),
        "records": records,
        "valid_records": valid_records,
        "invalid_records": records - valid_records,
        "duplicate_fingerprints": duplicate_fingerprints,
        "issues": issues,
        "candidate_only": True,
        "semantic_review_required": True,
        "ready_for_training": False,
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-domain candidate data from local Llama via Ollama")
    parser.add_argument("--output", default="data/llama_teacher_v1.jsonl")
    parser.add_argument("--per-domain", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit", metavar="JSONL", help="Audit an existing candidate dataset without generating data")
    args = parser.parse_args()
    if args.audit:
        report = audit_teacher_dataset(args.audit)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if report["invalid_records"]:
            raise SystemExit(1)
        raise SystemExit(0)
    os.environ.setdefault("TARA_LLM_BACKEND", "ollama")
    from src.llm_provider_factory import provider_from_env
    provider = provider_from_env()
    if provider is None:
        raise SystemExit("No teacher provider configured. Set TARA_LLM_BACKEND=ollama.")
    counts = generate_teacher_dataset(
        provider, args.output, per_domain=args.per_domain,
        max_completion_tokens=args.max_tokens, temperature=args.temperature,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": args.output, "counts": counts, "trained": False}, indent=2))


if __name__ == "__main__":
    main()
