from src.error_handling import (
    TARAExecutionError,
    TARAValidationError,
    exception_record,
    guarded,
    require,
)


def test_exception_record_is_structured():
    error = ValueError("bad input")
    record = exception_record(error, operation="unit-test", retryable=True)
    assert record.error_type == "ValueError"
    assert record.message == "bad input"
    assert record.operation == "unit-test"
    assert record.retryable is True


def test_guarded_preserves_cause_and_normalizes_failure():
    @guarded("unit-test")
    def fail():
        raise ValueError("boom")

    try:
        fail()
    except TARAExecutionError as exc:
        assert "unit-test failed" in str(exc)
        assert isinstance(exc.__cause__, ValueError)
    else:
        raise AssertionError("guarded function must raise")


def test_guarded_does_not_translate_tara_errors():
    @guarded("unit-test")
    def fail():
        raise TARAValidationError("invalid")

    try:
        fail()
    except TARAValidationError:
        pass
    else:
        raise AssertionError("TARA errors must remain typed")


def test_require_raises_typed_validation_error():
    try:
        require(False, "invalid state")
    except TARAValidationError as exc:
        assert str(exc) == "invalid state"
    else:
        raise AssertionError("require must raise")
