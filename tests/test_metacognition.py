from src.tara_mind.cognition.metacognition import MetacognitiveMonitor


def test_more_evidence_increases_confidence():
    monitor = MetacognitiveMonitor()
    low = monitor.assess(0.7, evidence_count=0)
    high = monitor.assess(0.7, evidence_count=5)
    assert high.confidence > low.confidence


def test_large_error_triggers_verification():
    report = MetacognitiveMonitor().assess(0.8, evidence_count=3, prediction_error=2.0)
    assert report.should_verify


def test_confidence_revision_is_signed():
    monitor = MetacognitiveMonitor()
    assert monitor.compare_revision(0.8, 0.5) < 0
