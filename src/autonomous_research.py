"""Autonomous research orchestration for TARA.

Research is evidence-gated and provenance-first. The engine decomposes a
question, collects source documents through an injected search provider,
extracts candidate evidence, cross-checks support/contradiction, produces
hypotheses with explicit uncertainty, and emits an auditable report.

No source is treated as ground truth merely because it was retrieved.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9_'-]+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class ResearchSource:
    source_id: str
    url: str
    title: str
    text: str
    authority: float = 0.5
    published_at: str = ""
    retrieved_at: str = ""
    metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class ResearchQuery:
    text: str
    reason: str
    priority: int = 0


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    queries: tuple[ResearchQuery, ...]


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    source_id: str
    claim: str
    snippet: str
    relevance: float
    authority: float


@dataclass(frozen=True)
class CrossCheckResult:
    claim: str
    supporting_sources: tuple[str, ...]
    contradicting_sources: tuple[str, ...]
    unresolved_sources: tuple[str, ...]
    confidence: float
    status: str


@dataclass(frozen=True)
class ResearchHypothesis:
    hypothesis_id: str
    statement: str
    evidence_ids: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class ResearchFinding:
    claim: str
    cross_check: CrossCheckResult
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ResearchReport:
    question: str
    plan: ResearchPlan
    sources: tuple[ResearchSource, ...]
    evidence: tuple[EvidenceItem, ...]
    findings: tuple[ResearchFinding, ...]
    hypotheses: tuple[ResearchHypothesis, ...]
    unresolved_questions: tuple[str, ...]
    overall_confidence: float
    fingerprint: str


class ResearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 5) -> Iterable[ResearchSource | Mapping[str, Any]]:
        ...


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token.lower() for token in _TOKEN_RE.findall(text))


def _fingerprint(parts: Iterable[str]) -> str:
    return sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _clip(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _jaccard(left: Sequence[str] | frozenset[str], right: Sequence[str] | frozenset[str]) -> float:
    a, b = set(left), set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class QueryDecomposer:
    """Create bounded research queries from one question."""

    def __init__(self, max_queries: int = 6):
        if max_queries <= 0:
            raise ValueError("max_queries must be positive")
        self.max_queries = max_queries

    def decompose(self, question: str) -> ResearchPlan:
        question = " ".join(str(question).split())
        if not question:
            raise ValueError("question must not be empty")

        parts = [part.strip(" ,;") for part in re.split(r"\s+(?:and|vs\.?|versus|compared with|while)\s+", question, flags=re.I)]
        parts = [part for part in parts if part]
        queries: list[ResearchQuery] = [ResearchQuery(question, "primary question", 100)]

        if len(parts) > 1:
            for index, part in enumerate(parts[: self.max_queries - 1], start=1):
                queries.append(ResearchQuery(part, f"decomposed sub-question {index}", 80 - index))

        if len(_tokens(question)) >= 6 and len(queries) < self.max_queries:
            queries.append(ResearchQuery(
                f"{question} evidence sources",
                "evidence-focused retrieval",
                50,
            ))
        return ResearchPlan(question, tuple(sorted(queries, key=lambda item: (-item.priority, item.text))))


class SourceNormalizer:
    """Normalize provider output while preserving provenance."""

    def normalize(self, item: ResearchSource | Mapping[str, Any], fallback_id: str) -> ResearchSource:
        if isinstance(item, ResearchSource):
            if not item.url and not item.source_id:
                raise ValueError("research source needs an id or url")
            return item

        if not isinstance(item, Mapping):
            raise TypeError("research provider results must be ResearchSource or mappings")

        url = str(item.get("url", "")).strip()
        title = str(item.get("title", "")).strip() or url or fallback_id
        text = str(item.get("text", item.get("content", "")))
        if not text.strip():
            raise ValueError("research source text must not be empty")
        source_id = str(item.get("source_id", "")).strip() or _fingerprint([url, title, text])[:24]
        metadata = tuple((str(k), str(v)) for k, v in sorted(item.get("metadata", {}).items())) if isinstance(item.get("metadata", {}), Mapping) else ()
        return ResearchSource(
            source_id=source_id,
            url=url,
            title=title,
            text=text[:100_000],
            authority=_clip(float(item.get("authority", 0.5))),
            published_at=str(item.get("published_at", "")),
            retrieved_at=str(item.get("retrieved_at", "")),
            metadata=metadata,
        )

    def deduplicate(self, items: Iterable[ResearchSource | Mapping[str, Any]]) -> tuple[ResearchSource, ...]:
        seen: set[str] = set()
        result: list[ResearchSource] = []
        for index, item in enumerate(items):
            source = self.normalize(item, f"source-{index}")
            key = source.url or source.source_id
            if key in seen:
                continue
            seen.add(key)
            result.append(source)
        return tuple(result)


class EvidenceExtractor:
    """Extract sentences that have measurable overlap with a research query."""

    def __init__(self, min_overlap: float = 0.12):
        if not 0.0 <= min_overlap <= 1.0:
            raise ValueError("min_overlap must be in [0, 1]")
        self.min_overlap = min_overlap

    def extract(self, query: str, sources: Sequence[ResearchSource]) -> tuple[EvidenceItem, ...]:
        query_tokens = _tokens(query)
        evidence: list[EvidenceItem] = []
        for source in sources:
            for sentence in _SENTENCE_RE.split(" ".join(source.text.split())):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                relevance = _jaccard(query_tokens, _tokens(sentence))
                if relevance < self.min_overlap:
                    continue
                evidence_id = _fingerprint([source.source_id, sentence])[:24]
                evidence.append(
                    EvidenceItem(
                        evidence_id=evidence_id,
                        source_id=source.source_id,
                        claim=sentence,
                        snippet=sentence[:600],
                        relevance=relevance,
                        authority=source.authority,
                    )
                )
        return tuple(sorted(evidence, key=lambda item: (-item.relevance * item.authority, item.evidence_id)))


class EvidenceCrossChecker:
    """Corroborate semantically similar evidence and detect lexical conflicts."""

    CONTRADICTIONS = (
        ({"is", "are", "was", "were"}, {"isn't", "aren't", "wasn't", "weren't", "not"}),
        ({"increase", "increases", "increased", "higher", "more"}, {"decrease", "decreases", "decreased", "lower", "less"}),
        ({"true", "valid", "supported"}, {"false", "invalid", "unsupported"}),
    )

    def compare(self, claim: str, evidence: Sequence[EvidenceItem]) -> CrossCheckResult:
        claim_tokens = _tokens(claim)
        supporting: list[tuple[str, float]] = []
        contradicting: list[tuple[str, float]] = []
        unresolved: list[tuple[str, float]] = []

        for item in evidence:
            overlap = _jaccard(claim_tokens, _tokens(item.claim))
            if overlap < 0.10:
                continue
            if self._contradicts(claim_tokens, _tokens(item.claim)):
                contradicting.append((item.source_id, overlap))
            elif overlap >= 0.28:
                supporting.append((item.source_id, overlap))
            else:
                unresolved.append((item.source_id, overlap))

        support_ids = tuple(sorted({source_id for source_id, _ in supporting}))
        contradict_ids = tuple(sorted({source_id for source_id, _ in contradicting}))
        unresolved_ids = tuple(sorted({source_id for source_id, _ in unresolved}))

        support_strength = min(1.0, 0.18 * len(support_ids) + (max((score for _, score in supporting), default=0.0) * 0.55))
        contradiction_penalty = min(0.9, 0.28 * len(contradict_ids))
        confidence = _clip(support_strength - contradiction_penalty)

        if contradict_ids and confidence < 0.5:
            status = "contradicted"
        elif len(support_ids) >= 2 and not contradict_ids:
            status = "corroborated"
        elif support_ids:
            status = "partially-supported"
        else:
            status = "unresolved"

        return CrossCheckResult(
            claim=claim,
            supporting_sources=support_ids,
            contradicting_sources=contradict_ids,
            unresolved_sources=unresolved_ids,
            confidence=confidence,
            status=status,
        )

    def _contradicts(self, left: frozenset[str], right: frozenset[str]) -> bool:
        for positive, negative in self.CONTRADICTIONS:
            if (left & positive and right & negative) or (right & positive and left & negative):
                return True
        return False


class HypothesisBuilder:
    def build(self, findings: Sequence[ResearchFinding]) -> tuple[ResearchHypothesis, ...]:
        hypotheses: list[ResearchHypothesis] = []
        for index, finding in enumerate(findings):
            check = finding.cross_check
            if check.status == "contradicted":
                continue
            statement = finding.claim
            confidence = _clip(check.confidence * (1.0 if check.status == "corroborated" else 0.8))
            hypothesis_id = _fingerprint([statement, *finding.evidence_ids])[:24]
            hypotheses.append(ResearchHypothesis(hypothesis_id, statement, finding.evidence_ids, confidence))
        return tuple(sorted(hypotheses, key=lambda item: (-item.confidence, item.hypothesis_id)))


class AutonomousResearchEngine:
    """End-to-end bounded research orchestration."""

    def __init__(
        self,
        *,
        decomposer: QueryDecomposer | None = None,
        normalizer: SourceNormalizer | None = None,
        extractor: EvidenceExtractor | None = None,
        checker: EvidenceCrossChecker | None = None,
        hypotheses: HypothesisBuilder | None = None,
        max_queries: int = 6,
        sources_per_query: int = 5,
    ):
        if sources_per_query <= 0:
            raise ValueError("sources_per_query must be positive")
        self.decomposer = decomposer or QueryDecomposer(max_queries)
        self.normalizer = normalizer or SourceNormalizer()
        self.extractor = extractor or EvidenceExtractor()
        self.checker = checker or EvidenceCrossChecker()
        self.hypotheses = hypotheses or HypothesisBuilder()
        self.sources_per_query = sources_per_query

    def research(
        self,
        question: str,
        searcher: Callable[[str], Iterable[ResearchSource | Mapping[str, Any]]] | ResearchProvider,
        *,
        max_queries: int | None = None,
    ) -> ResearchReport:
        plan = self.decomposer.decompose(question)
        queries = plan.queries[:max_queries] if max_queries is not None else plan.queries
        raw_sources: list[ResearchSource | Mapping[str, Any]] = []
        for query in queries:
            results = searcher.search(query.text, limit=self.sources_per_query) if hasattr(searcher, "search") else searcher(query.text)
            raw_sources.extend(list(results))

        sources = self.normalizer.deduplicate(raw_sources)
        evidence_by_query: dict[str, tuple[EvidenceItem, ...]] = {
            query.text: self.extractor.extract(query.text, sources) for query in queries
        }

        findings: list[ResearchFinding] = []
        for query in queries:
            evidence = evidence_by_query[query.text]
            if not evidence:
                findings.append(
                    ResearchFinding(
                        claim=query.text,
                        cross_check=CrossCheckResult(query.text, (), (), (), 0.0, "unresolved"),
                        evidence_ids=(),
                    )
                )
                continue
            best = evidence[0]
            check = self.checker.compare(best.claim, evidence)
            findings.append(
                ResearchFinding(
                    claim=best.claim,
                    cross_check=check,
                    evidence_ids=tuple(item.evidence_id for item in evidence[:8]),
                )
            )

        hypotheses = self.hypotheses.build(findings)
        unresolved = tuple(
            finding.claim for finding in findings if finding.cross_check.status in {"unresolved", "contradicted"}
        )
        overall = _clip(sum(h.confidence for h in hypotheses) / len(hypotheses)) if hypotheses else 0.0
        fingerprint = _fingerprint(
            [
                question,
                *(source.source_id for source in sources),
                *(hypothesis.hypothesis_id for hypothesis in hypotheses),
            ]
        )
        return ResearchReport(
            question=question,
            plan=ResearchPlan(plan.question, tuple(queries)),
            sources=sources,
            evidence=tuple(item for items in evidence_by_query.values() for item in items[:8]),
            findings=tuple(findings),
            hypotheses=hypotheses,
            unresolved_questions=unresolved,
            overall_confidence=overall,
            fingerprint=fingerprint,
        )
