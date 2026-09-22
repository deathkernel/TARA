from src.autonomous_research import (
    AutonomousResearchEngine,
    EvidenceCrossChecker,
    ResearchSource,
)


def test_research_pipeline_decomposes_collects_and_cross_checks():
    sources = [
        ResearchSource("a", "https://a.example", "A", "Python is widely used for data analysis and automation.", authority=0.9),
        ResearchSource("b", "https://b.example", "B", "Python is widely used for data analysis and automation.", authority=0.8),
        ResearchSource("c", "https://c.example", "C", "Some teams report lower adoption in embedded systems.", authority=0.7),
    ]

    def search(query):
        return sources

    report = AutonomousResearchEngine().research("Python and data analysis", search)
    assert len(report.plan.queries) >= 2
    assert len(report.sources) == 3
    assert report.hypotheses
    assert report.fingerprint


def test_research_is_explicitly_uncertain_without_evidence():
    report = AutonomousResearchEngine().research("unknown question", lambda query: [])
    assert report.overall_confidence == 0.0
    assert report.unresolved_questions


def test_contradiction_is_flagged():
    checker = EvidenceCrossChecker()
    from src.autonomous_research import EvidenceItem
    evidence = [
        EvidenceItem("1", "a", "The metric increased this year.", "The metric increased this year.", 0.8, 0.9),
        EvidenceItem("2", "b", "The metric decreased this year.", "The metric decreased this year.", 0.8, 0.9),
    ]
    result = checker.compare("The metric increased this year.", evidence)
    assert "b" in result.contradicting_sources
    assert result.status in {"contradicted", "partially-supported", "corroborated"}
