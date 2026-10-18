"""Logging helpers for autograd.py failures."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.autograd")

def log_autograd_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
