from src.checkpoint import load_checkpoint
from src.language_dataset import CausalTextDataset
from src.language_model import TinyLanguageModel
from train_tiny_lm import _batch_loss, _mean_loss


def test_batch_loss_is_token_weighted():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    batch = [([0, 1], [1, 2]), ([2], [3])]
    expected = (
        model.loss([0, 1], [1, 2]).data * 2
        + model.loss([2], [3]).data
    ) / 3
    assert abs(_batch_loss(model, batch).data - expected) < 1e-12


def test_validation_loss_does_not_change_parameters():
    model = TinyLanguageModel(vocab_size=4, embedding_dim=3, ff_dim=6, seed=2)
    dataset = CausalTextDataset([0, 1, 2, 3], context_length=2)
    before = [parameter.data for parameter in model.parameters()]
    value = _mean_loss(model, dataset, batch_size=2)
    after = [parameter.data for parameter in model.parameters()]
    assert value > 0.0
    assert after == before


def test_training_checkpoint_contains_final_metadata(tmp_path):
    from train_tiny_lm import train
    from src.tokenizer import CharTokenizer

    path = tmp_path / "final.json"
    train(steps=1, checkpoint_path=path)
    tokenizer = CharTokenizer("tara learns. tara reasons. ")
    state_model = TinyLanguageModel(
        vocab_size=tokenizer.vocab_size, embedding_dim=3, ff_dim=6, seed=7
    )
    state = load_checkpoint(state_model, path)
    assert state["step"] == 0
    assert "val_loss" in state["metrics"]
    assert state["scheduler"]["initial_lr"] == 0.03
