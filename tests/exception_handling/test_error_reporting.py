"""Tests for shared error reporting without swallowing exceptions."""

import logging
import pytest
from src.error_handling.common import report_exception


def test_report_exception_logs_operation_and_type(caplog: pytest.LogCaptureFixture) -> None:
    logger = logging.getLogger("tara.tests.error_reporting")
    error = ValueError("invalid checkpoint")
    with caplog.at_level(logging.ERROR, logger=logger.name):
        try:
            raise error
        except ValueError as caught:
            report_exception(logger, "load_checkpoint", caught)
    assert "load_checkpoint" in caplog.text
    assert "ValueError" in caplog.text
    assert "invalid checkpoint" in caplog.text
