from experiments.compare_trained_profiles import compare, validate_result


def test_trained_profiles_use_the_same_update_budget():
    results = compare(steps=1)
    assert results["tiny"]["config"] == results["scaled"]["config"]
    assert len(results["tiny"]["history"]) == 1
    assert len(results["scaled"]["history"]) == 1


def test_trained_profile_metrics_are_finite():
    results = compare(steps=1)
    validate_result(results)


def test_scaled_profile_has_more_capacity():
    results = compare(steps=1)
    assert results["scaled"]["parameters"] > results["tiny"]["parameters"]
