from experiments.compare_trained_profiles import compare, validate_result


# Compact but varied text prevents the BPE tokenizer from collapsing the test
# corpus into too few tokens for a train/validation split.
TEST_CORPUS = (
    "TARA builds reliable reasoning systems. "
    "Planning, memory, verification, learning, and tools work together. "
    "Experiments compare models using controlled budgets and finite metrics. "
    "Errors are logged, failures are bounded, and recovery remains explicit. "
) * 2


def test_trained_profiles_use_the_same_update_budget():
    results = compare(steps=1, corpus=TEST_CORPUS)
    assert results["tiny"]["config"] == results["scaled"]["config"]
    assert len(results["tiny"]["history"]) == 1
    assert len(results["scaled"]["history"]) == 1


def test_trained_profile_metrics_are_finite():
    results = compare(steps=1, corpus=TEST_CORPUS)
    validate_result(results)


def test_scaled_profile_has_more_capacity():
    results = compare(steps=1, corpus=TEST_CORPUS)
    assert results["scaled"]["parameters"] > results["tiny"]["parameters"]
