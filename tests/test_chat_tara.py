import torch

from scripts.chat.chat_tara import sample_token


def test_sample_token_returns_valid_id():
    logits = torch.tensor([0.0, 2.0, 1.0])
    token_id = sample_token(logits, __import__("random").Random(1), 1.0, 2, 1.0)
    assert token_id in {0, 1, 2}


def test_sample_token_rejects_invalid_temperature():
    try:
        sample_token(torch.tensor([1.0]), __import__("random").Random(1), 0.0, 1, 1.0)
    except ValueError:
        return
    raise AssertionError("temperature <= 0 should fail")
