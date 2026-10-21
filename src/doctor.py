"""Bounded TARA Doctor diagnostics and command scaffolding."""
from __future__ import annotations
import argparse, ast, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from .capability_registry import capabilities_as_json, inspect_capabilities
from .repository_audit import RepositoryAuditor
from .self_test import run_test_suite, summarize_pytest_output

@dataclass(frozen=True)
class DoctorFinding:
    severity: str
    code: str
    path: str
    line: int
    message: str
    suggestion: str

@dataclass(frozen=True)
class DoctorReport:
    root: str
    findings: tuple[DoctorFinding, ...]
    @property
    def healthy(self):
        return not any(item.severity == "ERROR" for item in self.findings)

COMMAND_CATALOG = {
    "check": "Static repository diagnostics; project code is not executed.",
    "capabilities": "Report source-file presence without importing code or claiming runtime readiness.",
    "tests": "Run the fixed pytest suite with a timeout and captured output.",
    "commands list": "List Doctor command groups.",
    "commands suggest": "Suggest diagnostics from a problem description.",
    "commands scaffold": "Create a review-required command skeleton.",
}
RULES = (
    (("syntax", "parse", "indentation", "syntaxerror"), "doctor check", "Run static syntax checks."),
    (("test", "regression", "pytest", "failing"), "doctor tests", "Run the fixed test suite and reproduce the failure."),
    (("import", "dependency", "module not found"), "doctor check", "Check imports and dependencies."),
    (("exception", "crash", "traceback", "error"), "doctor check", "Reproduce the failure and preserve its traceback."),
    (("slow", "hang", "timeout", "performance"), "new command: doctor profile", "Proposal: add bounded profiling with a timeout."),
    (("memory", "leak", "ram"), "new command: doctor memory", "Proposal: add opt-in resource diagnostics with strict limits."),
    (("security", "permission", "unsafe"), "new command: doctor security", "Proposal: add static permission-boundary checks."),
)

def _python_files(root: Path) -> Iterable[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}
    return (p for p in sorted(root.rglob("*.py")) if not any(part in ignored for part in p.parts))

def scan_repository(root="."):
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(str(base))
    findings = []
    audit = RepositoryAuditor(base).audit()
    for issue in audit.issues:
        # The detailed Python scan below reports syntax errors with a line number
        # and repair hint. Avoid emitting the auditor's less-detailed duplicate.
        if issue.code == "PYTHON_SYNTAX":
            continue
        severity = "WARNING" if issue.severity.lower() == "warning" else "ERROR"
        suggestion = ("Use a stable digest where values must persist." if issue.code == "NONDETERMINISTIC_HASH"
            else "Keep one canonical full-suite pytest workflow." if issue.code == "DUPLICATE_PYTEST_WORKFLOWS"
            else "Restore or update the required repository component." if issue.code == "MISSING_REQUIRED_FILE"
            else "Inspect the reported file and rerun Doctor after fixing it.")
        findings.append(DoctorFinding(severity, issue.code, issue.path, 0, issue.detail, suggestion))
    for path in _python_files(base):
        rel = str(path.relative_to(base))
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            findings.append(DoctorFinding("ERROR", "SOURCE_READ_FAILED", rel, 0, str(exc), "Check encoding and permissions."))
            continue
        try:
            tree = ast.parse(source, filename=rel)
        except SyntaxError as exc:
            findings.append(DoctorFinding(
                "ERROR",
                "PYTHON_SYNTAX",
                rel,
                exc.lineno or 0,
                exc.msg,
                "Fix the reported Python syntax error, then rerun Doctor.",
            ))
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "Exception" and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                findings.append(DoctorFinding("WARNING", "SILENT_BROAD_EXCEPTION", rel, node.lineno,
                    "except Exception: pass can hide failures.", "Log a redacted diagnostic or return an explicit failure result."))
        for number, line in enumerate(source.splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|BUG)\b", line, re.I):
                findings.append(DoctorFinding("INFO", "REVIEW_MARKER", rel, number, line.strip()[:240],
                    "Verify the marker and track real defects with a regression test."))
    findings.sort(key=lambda f: ({"ERROR": 0, "WARNING": 1, "INFO": 2}.get(f.severity, 3), f.path, f.line, f.code))
    return DoctorReport(str(base), tuple(findings))

def suggest_commands(description):
    query = description.strip().lower()
    if not query:
        raise ValueError("description must not be empty")
    matches = [(cmd, why) for words, cmd, why in RULES if any(word in query for word in words)]
    if not matches:
        matches = [("doctor check", "Start with static checks and add a focused regression test.")]
    output, seen = [], set()
    for item in matches:
        if item[0] not in seen:
            output.append(item); seen.add(item[0])
    return output

def scaffold_command(name, description, root="."):
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,48}", name):
        raise ValueError("name must be 2-49 lowercase letters, digits, or underscores")
    description = description.strip()
    if not description:
        raise ValueError("description must not be empty")
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(str(base))
    folder = base / "src" / "doctor_commands"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name + ".py")
    if target.exists():
        raise FileExistsError(str(target))
    safe = description.replace("\n", " ").strip()
    content = '"""Generated TARA Doctor command scaffold; review before use.\n"""\nDESCRIPTION = ' + repr(safe) + '\n\ndef run(root):\n    raise NotImplementedError("Implement ' + name + ' and add tests before registration")\n'
    target.write_text(content, encoding="utf-8")
    return target

def _print_report(report):
    counts = {level: sum(f.severity == level for f in report.findings) for level in ("ERROR", "WARNING", "INFO")}
    print(f"TARA Doctor: healthy={report.healthy} errors={counts['ERROR']} warnings={counts['WARNING']} info={counts['INFO']}")
    print("Repository:", report.root)
    for f in report.findings:
        where = f"{f.path}:{f.line}" if f.line else f.path
        print(f"{f.severity} {f.code} {where}: {f.message}\n  Suggestion: {f.suggestion}")
    if not report.findings:
        print("No findings from implemented checks; this does not prove the repository is bug-free.")
    return 0 if report.healthy else 1

def main(argv=None):
    parser = argparse.ArgumentParser(prog="tara doctor")
    sub = parser.add_subparsers(dest="action", required=True)
    check = sub.add_parser("check"); check.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    capabilities = sub.add_parser("capabilities"); capabilities.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1])); capabilities.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    tests = sub.add_parser("tests"); tests.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1])); tests.add_argument("--timeout", type=int, default=300)
    commands = sub.add_parser("commands"); cs = commands.add_subparsers(dest="command_action", required=True)
    cs.add_parser("list")
    sug = cs.add_parser("suggest"); sug.add_argument("description")
    scaffold = cs.add_parser("scaffold"); scaffold.add_argument("name"); scaffold.add_argument("--description", required=True); scaffold.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    if args.action == "check":
        return _print_report(scan_repository(args.root))
    if args.action == "capabilities":
        inventory = inspect_capabilities(args.root)
        if args.json:
            print(capabilities_as_json(inventory))
        else:
            print("TARA source capability inventory (source presence only; runtime activation is not verified)")
            for item in inventory:
                print(f"{item.status:16} {item.name} [{item.area}]")
                if item.present_files:
                    print("  present: " + ", ".join(item.present_files))
                if item.missing_files:
                    print("  missing: " + ", ".join(item.missing_files))
                print("  " + item.description)
        return 0
    if args.action == "tests":
        result = run_test_suite(args.root, timeout=args.timeout)
        print("Command:", " ".join(result.command))
        if result.stdout: print(result.stdout)
        if result.stderr: print(result.stderr, file=sys.stderr)
        summary = summarize_pytest_output(result.stdout, result.stderr)
        print("Test result:", "passed" if result.passed else "failed", {k: summary[k] for k in ("passed", "failed", "skipped", "errors")})
        return result.returncode
    if args.command_action == "list":
        for name, desc in COMMAND_CATALOG.items(): print(f"{name}: {desc}")
        return 0
    if args.command_action == "suggest":
        for command, reason in suggest_commands(args.description): print(f"{command}: {reason}")
        return 0
    if args.command_action == "scaffold":
        try: path = scaffold_command(args.name, args.description, args.root)
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            print("Cannot scaffold command:", exc, file=sys.stderr); return 2
        print("Created review-required scaffold:", path)
        print("Not registered or executed automatically; add tests and review before enabling.")
        return 0
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
