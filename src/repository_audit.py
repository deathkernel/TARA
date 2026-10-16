"""Static repository audit for TARA Phase 36.

The auditor is intentionally dependency-free. It checks Python syntax, required
core files, duplicate pytest workflows, and non-deterministic built-in hash()
calls. It never executes repository code.
"""
from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AuditIssue:
    code: str
    path: str
    detail: str
    severity: str = "error"


@dataclass(frozen=True)
class AuditReport:
    files_scanned: int
    python_files: int
    workflow_files: int
    issues: tuple[AuditIssue, ...]
    fingerprint: str

    @property
    def healthy(self) -> bool:
        return not self.issues


class RepositoryAuditor:
    REQUIRED_FILES = (
        "requirements.txt",
        "src/brain.py",
        "src/tara_core.py",
        "src/unified_cognitive_loop.py",
        "tests/test_tara_core.py",
    )
    RETIRED_TRAINING_PATHS = (
        "train_tiny_lm.py",
        "train_fast.py",
        "train_dataset.py",
        "train_algorithm_lm.py",
        "train_tinystories.py",
        "train_xor.py",
        "requirements-accelerated.txt",
        "src/language_model.py",
        "src/torch_language_model.py",
        "src/tokenizer.py",
        "src/continual_learning_engine.py",
        "src/training_pipeline.py",
        "src/checkpoint.py",
        "src/dataset_pipeline.py",
        "src/dataset_acquisition.py",
        "src/activations.py",
        "src/attention.py",
        "src/autograd.py",
        "src/embeddings.py",
        "src/generation.py",
        "src/gradcheck.py",
        "src/gradient_clipping.py",
        "src/initialization.py",
        "src/layers.py",
        "src/losses.py",
        "src/metrics.py",
        "src/mlp.py",
        "src/neuron.py",
        "src/numeric_inference.py",
        "src/optimizers.py",
        "src/positional.py",
        "src/schedulers.py",
        "src/transformer.py",
        "src/language_benchmarks.py",
        "src/evaluation.py",
        ".github/workflows/phase-b-intelligence.yml",
        ".github/workflows/phase-f-release-gate.yml",
        "docs/phase-b-intelligence.md",
    )

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(str(self.root))

    def audit(self) -> AuditReport:
        issues: list[AuditIssue] = []
        python_files = self._discover_python_files()
        workflow_files = sorted(
            path for path in (self.root / ".github" / "workflows").glob("*")
            if path.is_file()
        )

        for path in python_files:
            self._audit_python(path, issues)

        for relative in self.REQUIRED_FILES:
            if not (self.root / relative).is_file():
                issues.append(AuditIssue("MISSING_REQUIRED_FILE", relative, "required Phase 36 file is missing"))

        for relative in self.RETIRED_TRAINING_PATHS:
            if (self.root / relative).exists():
                issues.append(AuditIssue(
                    "RETIRED_TRAINING_SOURCE",
                    relative,
                    "local-model training, dataset, checkpoint, and tokenizer sources are retired; use the configured inference provider",
                ))

        requirements_path = self.root / "requirements.txt"
        if requirements_path.is_file():
            for line in requirements_path.read_text(encoding="utf-8").splitlines():
                requirement = line.strip().lower().split(";", 1)[0].split("[", 1)[0]
                if requirement == "torch" or requirement.startswith(("torch==", "torch>=", "torch<=", "torch>", "torch<", "torch!=", "torch~=")):
                    issues.append(AuditIssue(
                        "RETIRED_TRAINING_DEPENDENCY",
                        "requirements.txt",
                        "PyTorch is not needed for the provider-backed inference runtime",
                    ))
                    break

        pytest_workflows = []
        for path in workflow_files:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if "python -m pytest" in content:
                pytest_workflows.append(path)
        if len(pytest_workflows) > 1:
            names = ", ".join(str(path.relative_to(self.root)) for path in pytest_workflows)
            issues.append(AuditIssue(
                "DUPLICATE_PYTEST_WORKFLOWS",
                ".github/workflows",
                f"multiple workflows run the full pytest suite: {names}",
            ))

        parts = [str(issue.code) + "|" + issue.path + "|" + issue.detail for issue in issues]
        fingerprint = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return AuditReport(
            files_scanned=len(python_files) + len(workflow_files),
            python_files=len(python_files),
            workflow_files=len(workflow_files),
            issues=tuple(sorted(issues, key=lambda item: (item.code, item.path, item.detail))),
            fingerprint=fingerprint,
        )

    def _discover_python_files(self) -> list[Path]:
        """Discover Python sources while honoring Git's standard ignore rules.

        In a Git checkout, include tracked sources and non-ignored untracked
        sources. This keeps locally generated/ignored files out of the audit
        without hiding tracked files merely because an ignore rule was added.
        For temporary directories and source archives without Git metadata,
        fall back to filesystem discovery and the built-in directory exclusions.
        """
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(self.root), "ls-files",
                    "--cached", "--others", "--exclude-standard", "-z", "--", "*.py",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None

        if result is not None and result.returncode == 0:
            candidates = (
                self.root / relative
                for relative in result.stdout.split("\\0")
                if relative
            )
            return sorted(
                path for path in candidates
                if path.is_file() and path.suffix == ".py" and not self._ignored(path)
            )

        # Git may be unavailable or the root may be a non-repository directory.
        # Keep the audit usable in extracted source archives and isolated tests.
        return sorted(
            path for path in self.root.rglob("*.py")
            if not self._ignored(path)
        )

    def _audit_python(self, path: Path, issues: list[AuditIssue]) -> None:
        relative = str(path.relative_to(self.root))
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            issues.append(AuditIssue("PYTHON_SYNTAX", relative, str(exc)))
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "hash":
                issues.append(AuditIssue(
                    "NONDETERMINISTIC_HASH",
                    relative,
                    "built-in hash() is process-randomized; use stable hashing for persisted/fingerprint logic",
                    severity="warning",
                ))

    @staticmethod
    def _ignored(path: Path) -> bool:
        return any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the TARA static repository audit")
    parser.add_argument("root", nargs="?", default=".")
    args = parser.parse_args()
    report = RepositoryAuditor(args.root).audit()
    print(
        f"TARA audit: healthy={report.healthy} "
        f"python_files={report.python_files} workflow_files={report.workflow_files} "
        f"issues={len(report.issues)} fingerprint={report.fingerprint}"
    )
    for issue in report.issues:
        print(f"{issue.severity.upper()} {issue.code} {issue.path}: {issue.detail}")
    return 0 if report.healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
