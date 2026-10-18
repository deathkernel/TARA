"""Logging helpers for tara_core.py lifecycle failures."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.core")

def log_core_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
