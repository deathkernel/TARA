from pathlib import Path


def test_learning_experiment_cli_exists():
    path = Path("run_learning_experiment.py")
    assert path.exists()
    source = path.read_text(encoding="utf-8")
    assert "TrainingExperiment" in source
    assert "LearningExperiment" in source
    assert "report.decision.accepted" in source
