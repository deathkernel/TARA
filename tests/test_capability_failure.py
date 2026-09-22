from src.capability_failure import CapabilityFailureAnalyzer
from src.capability_suite import capability_cases
from src.model_capability_runner import ModelEvaluation
from src.intelligence_benchmark import IntelligenceBenchmark


def test_failure_analyzer_records_observations_and_prioritizes_categories():
    cases = capability_cases()
    observations = {case.case_id: "wrong" for case in cases}
    report = IntelligenceBenchmark("tara-capability-v1", cases).run(observations.__getitem__)
    evaluation = ModelEvaluation("checkpoint.pt", report, "model", "evaluation", observations)

    diagnosis = CapabilityFailureAnalyzer().diagnose(evaluation, cases)

    assert len(diagnosis.failures) == 21
    assert len(diagnosis.targets) == 7
    assert diagnosis.targets[0].priority == diagnosis.targets[0].failed_cases
    assert all(item.kind == "incorrect_answer" for item in diagnosis.failures)


def test_failure_analyzer_handles_empty_output():
    cases = capability_cases()
    observations = {case.case_id: "" for case in cases}
    report = IntelligenceBenchmark("tara-capability-v1", cases).run(observations.__getitem__)
    evaluation = ModelEvaluation("checkpoint.pt", report, "model", "evaluation", observations)

    diagnosis = CapabilityFailureAnalyzer().diagnose(evaluation, cases)

    assert {item.kind for item in diagnosis.failures} == {"empty_output"}
