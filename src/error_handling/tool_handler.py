"""Logging helpers for tool execution failures."""

import logging
from src.error_handling.common import report_exception

logger = logging.getLogger("tara.tools")

def log_tool_error(operation: str, error: BaseException) -> None:
    report_exception(logger, operation, error)
