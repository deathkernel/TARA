"""Safe algorithm invention and verification loop for TARA.

This module lets TARA *invent algorithms* as structured, human-readable
proposals instead of treating generated source code as truth. The loop is:

problem framing -> diverse strategy generation -> invariant/complexity review
-> external verification -> counterexample feedback -> refinement -> promotion.

Verification is injected by the caller. This module never executes generated
code and never mutates the repository or production runtime automatically.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence


@dataclass(frozen=True)
class ProblemFrame:
    """A compact formal frame TARA can reason over before inventing a solution."""

    problem: str
    objective: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    invariants: tuple[str, ...] = ()
    success_criteria: tuple[str, ...] = ()


@dataclass(frozen=True)
class AlgorithmProposal:
    """A non-executable algorithm hypothesis."""

    name: str
    strategy: str
    idea: str
    steps: tuple[str, ...]
    invariant: str
    correctness_argument: str
    time_complexity: str
    space_complexity: str
    assumptions: tuple[str, ...] = ()
    parent: str | None = None

    @property
    def fingerprint(self) -> str:
        payload = json.dumps({
            "name": self.name, "strategy": self.strategy, "idea": self.idea,
            "steps": self.steps, "invariant": self.invariant,
            "correctness_argument": self.correctness_argument,
            "time_complexity": self.time_complexity,
            "space_complexity": self.space_complexity,
            "assumptions": self.assumptions,
        }, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class VerificationReport:
    """Evidence supplied by an external verifier."""

    candidate: AlgorithmProposal
    verified: bool
    score: float | None
    feedback: str = ""
    counterexamples: tuple[str, ...] = ()
    tests_passed: int = 0
    tests_total: int = 0

    @property
    def correctness(self) -> float:
        if self.tests_total > 0:
            return self.tests_passed / self.tests_total
        return 1.0 if self.verified else 0.0


@dataclass(frozen=True)
class DiscoveryCycle:
    frame: ProblemFrame
    reports: tuple[VerificationReport, ...]
    best: AlgorithmProposal | None
    rounds: int


class StrategyLibrary:
    """Diverse algorithmic lenses used to avoid repeating one family."""

    DEFAULT = (
        "brute_force_with_pruning",
        "divide_and_conquer",
        "greedy_with_invariant",
        "dynamic_programming",
        "two_pointers_or_sliding_window",
        "hashing_or_indexing",
        "binary_search_on_answer",
        "graph_traversal_or_shortest_path",
        "sweep_line_or_sorting",
        "bitwise_or_mathematical",
        "randomized_or_sampling",
    )

    def __init__(self, strategies: Sequence[str] | None = None) -> None:
        values = tuple(str(x).strip() for x in (strategies or self.DEFAULT) if str(x).strip())
        if not values:
            raise ValueError("strategy library must not be empty")
        self.strategies = values

    def choose(self, count: int, *, offset: int = 0) -> tuple[str, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        return tuple(self.strategies[(offset + index) % len(self.strategies)] for index in range(count))


class AlgorithmArchive:
    """Append-only JSONL memory for proposed and verified algorithm knowledge."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def save(self, report: VerificationReport) -> bool:
        existing = {item.get("fingerprint") for item in self._records()}
        if report.candidate.fingerprint in existing:
            return False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "fingerprint": report.candidate.fingerprint,
            "candidate": asdict(report.candidate),
            "verification": {
                "verified": report.verified, "score": report.score,
                "feedback": report.feedback,
                "counterexamples": list(report.counterexamples),
                "tests_passed": report.tests_passed,
                "tests_total": report.tests_total,
            },
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return True

    def _records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid algorithm archive JSON at line {line_number}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"invalid algorithm archive record at line {line_number}")
                records.append(item)
        return records

    def history(self, problem: str | None = None) -> tuple[dict, ...]:
        records = self._records()
        if problem is None:
            return tuple(records)
        needle = problem.strip().lower()
        return tuple(
            item for item in records
            if needle in item.get("candidate", {}).get("idea", "").lower()
            or needle in item.get("candidate", {}).get("name", "").lower()
        )


def frame_problem(problem: str) -> ProblemFrame:
    """Extract useful structure without pretending heuristic parsing is proof."""

    text = problem.strip()
    if not text:
        raise ValueError("problem must not be empty")
    sentences = tuple(part.strip(" .") for part in re.split(r"[.;]", text) if part.strip())
    constraints = tuple(
        sentence for sentence in sentences
        if re.search(r"\b(must|cannot|only|at most|at least|limit|constraint)\b", sentence, re.I)
    )
    return ProblemFrame(
        problem=text,
        objective=text,
        constraints=constraints,
        success_criteria=("correct for all valid inputs", "respect stated constraints"),
    )


def _similarity(left: AlgorithmProposal, right: AlgorithmProposal) -> float:
    def tokens(item: AlgorithmProposal) -> set[str]:
        text = " ".join((item.strategy, item.idea, *item.steps, item.invariant)).lower()
        return set(re.findall(r"[a-z0-9_]+", text))
    a, b = tokens(left), tokens(right)
    return len(a & b) / max(1, len(a | b))


class AlgorithmDiscoveryEngine:
    """Iterative, evidence-gated algorithm invention engine.

    The generator proposes structured hypotheses. The verifier is the authority
    for correctness. A candidate is never promoted merely because a model liked
    its explanation.
    """

    def __init__(
        self,
        generator: Callable[[ProblemFrame, str, tuple[str, ...], int], Iterable[AlgorithmProposal]],
        verifier: Callable[[ProblemFrame, AlgorithmProposal], VerificationReport],
        *,
        archive: AlgorithmArchive | None = None,
        strategies: StrategyLibrary | None = None,
        max_candidates_per_round: int = 8,
    ) -> None:
        if max_candidates_per_round < 1:
            raise ValueError("max_candidates_per_round must be positive")
        self.generator = generator
        self.verifier = verifier
        self.archive = archive
        self.strategies = strategies or StrategyLibrary()
        self.max_candidates_per_round = max_candidates_per_round

    def discover(self, problem: str | ProblemFrame, *, rounds: int = 4, candidates_per_round: int | None = None) -> DiscoveryCycle:
        if rounds < 1:
            raise ValueError("rounds must be positive")
        frame = problem if isinstance(problem, ProblemFrame) else frame_problem(problem)
        count = min(candidates_per_round or self.max_candidates_per_round, self.max_candidates_per_round)
        if count < 1:
            raise ValueError("candidates_per_round must be positive")

        reports: list[VerificationReport] = []
        feedback: tuple[str, ...] = ()
        best: VerificationReport | None = None

        for round_index in range(rounds):
            selected_strategies = self.strategies.choose(count, offset=round_index * count)
            generated = list(self.generator(frame, "\n".join(selected_strategies), feedback, count))
            candidates = self._deduplicate(generated, count)
            if not candidates:
                break

            round_reports: list[VerificationReport] = []
            for candidate in candidates:
                report = self.verifier(frame, candidate)
                round_reports.append(report)
                reports.append(report)
                if self.archive is not None:
                    self.archive.save(report)

            round_best = self._best(round_reports)
            if round_best is not None and (best is None or self._better(round_best, best)):
                best = round_best

            failures = []
            for report in round_reports:
                if not report.verified:
                    failures.extend(report.counterexamples[:3])
                    if report.feedback:
                        failures.append(report.feedback)
            feedback = tuple(dict.fromkeys(failures))[:12]
            if round_best is not None and round_best.verified:
                feedback = (
                    f"Verified candidate score={round_best.score!r}; seek a correct or faster variant.",
                    *feedback,
                )

        return DiscoveryCycle(frame, tuple(reports), None if best is None else best.candidate, rounds if reports else 0)

    @staticmethod
    def _deduplicate(candidates: Iterable[AlgorithmProposal], limit: int) -> list[AlgorithmProposal]:
        seen: set[str] = set()
        output: list[AlgorithmProposal] = []
        for candidate in candidates:
            if not isinstance(candidate, AlgorithmProposal):
                continue
            if not candidate.name.strip() or not candidate.steps or not candidate.invariant.strip():
                continue
            if candidate.fingerprint in seen:
                continue
            seen.add(candidate.fingerprint)
            if any(_similarity(candidate, other) >= 0.92 for other in output):
                continue
            output.append(candidate)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _best(reports: Sequence[VerificationReport]) -> VerificationReport | None:
        return min(reports, key=AlgorithmDiscoveryEngine._sort_key, default=None)

    @staticmethod
    def _sort_key(report: VerificationReport):
        return (
            0 if report.verified else 1,
            -report.correctness,
            float("inf") if report.score is None else report.score,
            len(report.candidate.steps),
        )

    @staticmethod
    def _better(left: VerificationReport, right: VerificationReport) -> bool:
        return AlgorithmDiscoveryEngine._sort_key(left) < AlgorithmDiscoveryEngine._sort_key(right)


def proposal_from_dict(data: dict) -> AlgorithmProposal:
    """Strictly parse model output into an algorithm hypothesis."""

    required = (
        "name", "strategy", "idea", "steps", "invariant",
        "correctness_argument", "time_complexity", "space_complexity",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError("missing algorithm fields: " + ", ".join(missing))
    steps = tuple(str(item).strip() for item in data["steps"] if str(item).strip())
    assumptions = tuple(str(item).strip() for item in data.get("assumptions", ()) if str(item).strip())
    if not steps:
        raise ValueError("algorithm must contain at least one step")
    return AlgorithmProposal(
        name=str(data["name"]).strip(),
        strategy=str(data["strategy"]).strip(),
        idea=str(data["idea"]).strip(),
        steps=steps,
        invariant=str(data["invariant"]).strip(),
        correctness_argument=str(data["correctness_argument"]).strip(),
        time_complexity=str(data["time_complexity"]).strip(),
        space_complexity=str(data["space_complexity"]).strip(),
        assumptions=assumptions,
        parent=str(data["parent"]).strip() if data.get("parent") else None,
    )


def build_algorithm_prompt(frame: ProblemFrame, strategy: str, feedback: Sequence[str] = ()) -> str:
    feedback_text = "\n".join(f"- {item}" for item in feedback) or "- none"
    fence = chr(96) * 3
    return (
        "TARA ALGORITHM INVENTION TASK\n"
        "Invent a genuinely different algorithmic hypothesis; do not merely rename a known one.\n"
        f"Problem: {frame.problem}\n"
        f"Objective: {frame.objective}\n"
        f"Constraints: {frame.constraints or ('none stated',)}\n"
        f"Preferred strategy lens: {strategy}\n"
        f"Previous verification feedback:\n{feedback_text}\n\n"
        "Return JSON with exactly these fields: "
        "name, strategy, idea, steps, invariant, correctness_argument, "
        "time_complexity, space_complexity, assumptions, parent. "
        "steps and assumptions must be arrays of strings. "
        "State an invariant and a concrete correctness argument. "
        "Do not output executable source code. "
        f"Example format: {fence}json ... {fence}"
    )


class ModelAlgorithmGenerator:
    """Turn a text generator into a structured algorithm-hypothesis generator."""

    def __init__(self, generate_text: Callable[..., str], *, max_new_tokens: int = 384, temperature: float = 0.8) -> None:
        if max_new_tokens <= 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be positive and temperature must be positive")
        self.generate_text = generate_text
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, frame: ProblemFrame, strategy_text: str, feedback: tuple[str, ...], count: int) -> Iterable[AlgorithmProposal]:
        strategies = [item.strip() for item in strategy_text.splitlines() if item.strip()]
        for strategy in strategies[:count]:
            prompt = build_algorithm_prompt(frame, strategy, feedback)
            raw = self.generate_text(prompt, max_new_tokens=self.max_new_tokens, temperature=self.temperature)
            data = self._parse_json(raw)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        yield proposal_from_dict(item)
            elif isinstance(data, dict):
                yield proposal_from_dict(data)

    @staticmethod
    def _parse_json(raw: str):
        text = raw.strip()
        fence = chr(96) * 3
        fenced = re.search(f"{re.escape(fence)}(?:json)?\\s*(\\{{.*?\\}}|\\[.*?\\])\\s*{re.escape(fence)}", text, re.S | re.I)
        if fenced:
            text = fenced.group(1)
        else:
            match = re.search(r"(\{.*\}|\[.*\])", text, re.S)
            if match:
                text = match.group(1)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("model did not return valid algorithm JSON") from exc
