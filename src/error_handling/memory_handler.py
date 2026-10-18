"""Logging helpers for memory/retrieval failures in the cognitive loop."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.memory")

def log_memory_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
