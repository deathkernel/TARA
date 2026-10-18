"""Logging helpers for basic_brain.py failures."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.brain")

def log_brain_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
