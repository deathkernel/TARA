"""Polyglot algorithm planning, candidates, execution, and improvement.

The model-backed adapter is exposed lazily so importing algorithm/novelty
utilities does not require PyTorch to load native DLLs.
"""

from .archive import CandidateArchive
from .benchmark import BenchmarkResult, PolyglotBenchmark, rank_benchmarks
from .candidate import PolyglotCandidate
from .execution import PolyglotExecutionError, PolyglotExecutor
from .generation import ModelCandidateGenerator, build_candidate_prompt
from .knowledge import KnowledgeExtractor, KnowledgeRecord
from .languages import LanguageSpec, default_languages, select_language
from .learning import LearningExport, VerifiedKnowledgeLearner
from .novelty import NoveltyAnalyzer, NoveltyReport, SimilarityMatch, similarity, structural_signature
from .parser import CandidateParseError, parse_candidate
from .result import ExecutionResult
from .self_improvement import ImprovementResult, SelfImprovementEngine

__all__ = [
    "BenchmarkResult", "CandidateArchive", "CandidateParseError", "ExecutionResult",
    "ImprovementResult", "KnowledgeExtractor", "KnowledgeRecord", "LanguageSpec",
    "ModelCandidateGenerator", "NoveltyAnalyzer", "NoveltyReport",
    "PolyglotBenchmark", "PolyglotCandidate", "PolyglotExecutionError", "PolyglotExecutor",
    "SelfImprovementEngine", "SimilarityMatch", "TARAAlgorithmModel",
    "VerifiedKnowledgeLearner", "build_candidate_prompt", "default_languages",
    "parse_candidate", "rank_benchmarks", "select_language", "similarity",
    "structural_signature",
]


def __getattr__(name: str):
    """Load the neural model adapter only when it is explicitly requested."""
    if name == "TARAAlgorithmModel":
        from .model_adapter import TARAAlgorithmModel
        return TARAAlgorithmModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
