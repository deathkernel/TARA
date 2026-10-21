"""Static repository audit for TARA Phase 36.

The dependency-free auditor checks Python syntax, required core files, duplicate
pytest workflows, non-deterministic built-in hash() calls, and common .gitignore
rules. It never executes repository code.
"""
from __future__ import annotations

import ast
import fnmatch
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


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
    ALWAYS_IGNORED = {".git", ".venv", "venv", "__pycache__"}

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(str(self.root))

    def audit(self) -> AuditReport:
        issues: list[AuditIssue] = []
        python_files = sorted(
            path for path in self.root.rglob("*.py") if not self._ignored(path)
        )
        workflow_dir = self.root / ".github" / "workflows"
        workflow_files = sorted(
            path for path in workflow_dir.glob("*")
            if path.is_file() and not self._ignored(path)
        ) if workflow_dir.is_dir() else []

        for path in python_files:
            self._audit_python(path, issues)

        for relative in self.REQUIRED_FILES:
            if not (self.root / relative).is_file():
                issues.append(AuditIssue(
                    "MISSING_REQUIRED_FILE", relative,
                    "required Phase 36 file is missing",
                ))

        pytest_workflows = []
        for path in workflow_files:
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "python -m pytest" in content:
                pytest_workflows.append(path)
        if len(pytest_workflows) > 1:
            names = ", ".join(
                str(path.relative_to(self.root)) for path in pytest_workflows
            )
            issues.append(AuditIssue(
                "DUPLICATE_PYTEST_WORKFLOWS", ".github/workflows",
                f"multiple workflows run the full pytest suite: {names}",
            ))

        parts = [
            f"{issue.code}|{issue.path}|{issue.detail}" for issue in issues
        ]
        fingerprint = sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        return AuditReport(
            files_scanned=len(python_files) + len(workflow_files),
            python_files=len(python_files),
            workflow_files=len(workflow_files),
            issues=tuple(sorted(
                issues, key=lambda item: (item.code, item.path, item.detail)
            )),
            fingerprint=fingerprint,
        )

    def _audit_python(self, path: Path, issues: list[AuditIssue]) -> None:
        relative = path.relative_to(self.root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            issues.append(AuditIssue("PYTHON_SYNTAX", relative, str(exc)))
            return
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                issues.append(AuditIssue(
                    "NONDETERMINISTIC_HASH", relative,
                    "built-in hash() is process-randomized; use stable hashing for persisted/fingerprint logic",
                    severity="warning",
                ))

    def _ignored(self, path: Path) -> bool:
        """Respect built-in exclusions and common root/nested .gitignore rules.

        Supports comments, negation, directory-only rules, slash-containing
        patterns, and * / ? wildcards; this is not the full Git pathspec engine.
        """
        try:
            path = path.resolve()
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            return True
        if any(part in self.ALWAYS_IGNORED for part in Path(relative).parts):
            return True

        ignored = False
        for base, pattern, negated, directory_only, anchored in self._ignore_rules(path):
            try:
                candidate = path.relative_to(base).as_posix()
            except ValueError:
                continue
            if self._matches_rule(pattern, candidate, directory_only, anchored):
                ignored = not negated
        return ignored

    def _ignore_rules(self, path: Path):
        directories = [self.root]
        parents = []
        current = path.parent
        while current != self.root and self.root in current.parents:
            parents.append(current)
            current = current.parent
        directories.extend(reversed(parents))
        for directory in directories:
            ignore_file = directory / ".gitignore"
            if not ignore_file.is_file():
                continue
            try:
                lines = ignore_file.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for raw in lines:
                line = raw.rstrip()
                if not line:
                    continue
                if line.startswith(r"\#"):
                    line = line[1:]
                elif line.startswith("#"):
                    continue
                negated = line.startswith("!")
                if negated:
                    line = line[1:]
                if line.startswith(r"\!"):
                    line = line[1:]
                    negated = False
                directory_only = line.endswith("/")
                anchored = line.startswith("/")
                line = line.strip("/")
                if line:
                    yield directory, line, negated, directory_only, anchored

    @staticmethod
    def _matches_rule(
        pattern: str, candidate: str, directory_only: bool, anchored: bool = False
    ) -> bool:
        parts = candidate.split("/")
        prefixes = ["/".join(parts[:i]) for i in range(1, len(parts) + 1)]
        if anchored or "/" in pattern:
            prefixes = prefixes[:1] if anchored else prefixes
        for prefix in prefixes:
            if directory_only and prefix == candidate:
                continue  # The scanned target is a file; only ancestor dirs match.
            if "/" not in pattern and not anchored:
                if fnmatch.fnmatchcase(prefix.rsplit("/", 1)[-1], pattern):
                    return True
            elif _slash_glob_match(pattern, prefix):
                return True
        return False


def _slash_glob_match(pattern: str, value: str) -> bool:
    """Match slash-aware globs where * and ? cannot cross path separators."""
    pieces = ["^"]
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*" and index + 1 < len(pattern) and pattern[index + 1] == "*":
            pieces.append(".*")
            index += 2
            continue
        if char == "*":
            pieces.append("[^/]*")
        elif char == "?":
            pieces.append("[^/]")
        elif char == "[":
            end = pattern.find("]", index + 1)
            if end != -1:
                pieces.append(pattern[index:end + 1])
                index = end
            else:
                pieces.append(r"\[")
        else:
            pieces.append(re.escape(char))
        index += 1
    pieces.append("$")
    return re.match("".join(pieces), value) is not None


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
