"""Polyglot algorithm planning, candidate generation, execution, and improvement."""

from .archive import CandidateArchive
from .benchmark import BenchmarkResult, PolyglotBenchmark, rank_benchmarks
from .candidate import PolyglotCandidate
from .execution import PolyglotExecutionError, PolyglotExecutor
from .generation import ModelCandidateGenerator, build_candidate_prompt
from .knowledge import KnowledgeExtractor, KnowledgeRecord
from .languages import LanguageSpec, default_languages, select_language
from .novelty import NoveltyAnalyzer, NoveltyReport, SimilarityMatch, similarity, structural_signature
from .parser import CandidateParseError, parse_candidate
from .result import ExecutionResult
from .self_improvement import ImprovementResult, SelfImprovementEngine

__all__ = [
    "BenchmarkResult", "CandidateArchive", "CandidateParseError", "ExecutionResult", "ImprovementResult",
    "KnowledgeExtractor", "KnowledgeRecord", "LanguageSpec", "ModelCandidateGenerator",
    "NoveltyAnalyzer", "NoveltyReport", "PolyglotCandidate", "PolyglotExecutionError",
    "PolyglotBenchmark", "PolyglotExecutor", "SelfImprovementEngine", "SimilarityMatch", "build_candidate_prompt",
    "default_languages", "parse_candidate", "rank_benchmarks", "select_language", "similarity", "structural_signature",
]
