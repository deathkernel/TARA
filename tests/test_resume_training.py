from train_tiny_lm import train
from src.checkpoint import load_checkpoint


def test_resume_matches_uninterrupted_training(tmp_path):
    full_path = tmp_path / "full.json"
    split_path = tmp_path / "split.json"

    full_model, _ = train(steps=8, total_steps=8, checkpoint_path=full_path)
    train(steps=4, total_steps=8, checkpoint_path=split_path)
    resumed_model, _ = train(
        steps=4,
        total_steps=8,
        resume_from=split_path,
        checkpoint_path=split_path,
    )

    full_values = [parameter.data for parameter in full_model.parameters()]
    resumed_values = [parameter.data for parameter in resumed_model.parameters()]
    assert resumed_values == full_values

    state = load_checkpoint(resumed_model, split_path)
    assert state["step"] == 7


def test_resume_rejects_different_schedule_horizon(tmp_path):
    path = tmp_path / "tara.json"
    train(steps=3, total_steps=8, checkpoint_path=path)

    try:
        train(steps=1, total_steps=9, resume_from=path)
    except ValueError as exc:
        assert "total_steps" in str(exc)
    else:
        raise AssertionError("resume should reject a different schedule horizon")
