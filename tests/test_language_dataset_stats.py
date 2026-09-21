from src.language_dataset import CausalTextDataset, dataset_statistics, token_frequency


def test_dataset_statistics_reports_tokens_windows_and_unknown_rate():
    dataset = CausalTextDataset([0, 1, 1, 2, 0], context_length=3)
    stats = dataset_statistics(dataset, unk_id=0)
    assert stats["token_count"] == 5
    assert stats["unique_tokens"] == 3
    assert stats["window_count"] == 4
    assert stats["context_length"] == 3
    assert stats["unknown_tokens"] == 2
    assert stats["unknown_rate"] == 0.4


def test_token_frequency_is_deterministic_and_limited():
    assert token_frequency([2, 1, 2, 3, 1, 2], limit=2) == [(2, 3), (1, 2)]
