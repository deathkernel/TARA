from experiments.scientific_scaling import run_scaling_experiment


def test_scaling_experiment_uses_matched_budget_and_reports_finite_metrics():
    report = run_scaling_experiment(steps=1)
    assert report["protocol"]["steps"] == 1
    assert report["protocol"]["optimizer"] == "AdamW"
    assert {item["profile"] for item in report["profiles"]} == {"tiny", "scaled"}
    for item in report["profiles"]:
        assert item["parameters"] > 0
        assert item["updates"] == 1
        assert item["token_budget_proxy"] == item["tokens_per_update_proxy"]
        assert item["train_loss_final"] == item["train_loss_final"]
        assert item["validation_loss_final"] == item["validation_loss_final"]
        assert item["heldout_loss"] == item["heldout_loss"]


def test_scaling_experiment_rejects_nonpositive_budget():
    try:
        run_scaling_experiment(steps=0)
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive budget should fail")
