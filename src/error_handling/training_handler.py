"""Logging helpers for training_pipeline.py failures."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.training")

def log_training_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
