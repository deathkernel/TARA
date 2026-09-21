import pytest

from src.language_dataset import (
    CausalTextDataset,
    build_causal_datasets,
    build_train_validation_datasets,
    split_token_ids,
)
from src.tokenizer import BPETokenizer, CharTokenizer


def test_causal_dataset_returns_next_token_pairs():
    dataset = CausalTextDataset([0, 1, 2, 3, 4], context_length=3)
    assert len(dataset) == 4
    assert dataset[0] == ([0, 1, 2], [1, 2, 3])
    assert dataset[3] == ([3], [4])


def test_iter_windows_covers_stream_without_dropping_tail():
    dataset = CausalTextDataset(list(range(7)), context_length=3)
    assert list(dataset.iter_windows()) == [
        ([0, 1, 2], [1, 2, 3]),
        ([3, 4, 5], [4, 5, 6]),
    ]


def test_batches_have_requested_size_and_cover_all_examples():
    dataset = CausalTextDataset(list(range(8)), context_length=2)
    batches = list(dataset.iter_batches(batch_size=3))
    assert [len(batch) for batch in batches] == [3, 3, 1]
    flattened = [item for batch in batches for item in batch]
    assert flattened == [dataset[index] for index in range(len(dataset))]


def test_batch_at_matches_iterator():
    dataset = CausalTextDataset(list(range(12)), context_length=3)
    batches = list(dataset.iter_batches(batch_size=4, shuffle=True, seed=11))
    assert dataset.batch_count(4) == 3
    for index, batch in enumerate(batches):
        assert dataset.batch_at(index, 4, shuffle=True, seed=11) == batch


def test_shuffled_batches_are_deterministic():
    dataset = CausalTextDataset(list(range(8)), context_length=2)
    first = list(dataset.iter_batches(batch_size=3, shuffle=True, seed=11))
    second = list(dataset.iter_batches(batch_size=3, shuffle=True, seed=11))
    assert first == second
    assert first != list(dataset.iter_batches(batch_size=3, shuffle=True, seed=12))


def test_batch_rejects_invalid_size():
    dataset = CausalTextDataset([1, 2, 3], context_length=2)
    with pytest.raises(ValueError):
        list(dataset.iter_batches(batch_size=0))
    with pytest.raises(TypeError):
        list(dataset.iter_batches(batch_size=1.5))


def test_dataset_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        CausalTextDataset([1], context_length=2)
    with pytest.raises(ValueError):
        CausalTextDataset([1, 2], context_length=0)
    with pytest.raises(IndexError):
        CausalTextDataset([1, 2], context_length=2)[2]


def test_split_preserves_order_and_is_deterministic():
    train, validation = split_token_ids(list(range(20)), validation_fraction=0.2)
    assert train == list(range(16))
    assert validation == list(range(16, 20))


def test_split_rejects_invalid_fraction():
    with pytest.raises(ValueError):
        split_token_ids([0, 1, 2, 3], validation_fraction=0.0)
    with pytest.raises(ValueError):
        split_token_ids([0, 1, 2, 3], validation_fraction=1.0)


def test_build_causal_datasets_uses_tokenizer_once():
    tokenizer = CharTokenizer("abcdefghij")
    train, validation = build_causal_datasets(
        tokenizer,
        "abcdefghij",
        context_length=3,
        validation_fraction=0.2,
    )
    assert train.token_ids == tokenizer.encode("abcdefgh")
    assert validation.token_ids == tokenizer.encode("ij")


def test_train_validation_pipeline_fits_tokenizer_only_on_train_text():
    tokenizer, train, validation = build_train_validation_datasets(
        lambda text: BPETokenizer(text, vocab_size=32),
        "aaaaabbbbbcccccdddddeeeee",
        context_length=2,
        validation_fraction=0.2,
    )
    assert tokenizer.encode("x") == [tokenizer.stoi[tokenizer.UNK]]
    assert train.token_ids
    assert validation.token_ids
