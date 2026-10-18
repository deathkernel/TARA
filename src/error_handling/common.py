"""Shared primitives used by module-specific error handlers."""

from __future__ import annotations

import logging
from types import TracebackType


def report_exception(
    logger: logging.Logger,
    operation: str,
    error: BaseException,
    *,
    traceback: TracebackType | None = None,
) -> None:
    """Log an exception with traceback without swallowing or transforming it.

    Call this inside an except block and use a bare raise when the error must
    propagate. Arbitrary payloads are intentionally not logged to avoid leaking
    prompts, credentials, or user data.
    """
    logger.error(
        "TARA operation failed: %s (error_type=%s)",
        operation,
        type(error).__name__,
        exc_info=(type(error), error, traceback if traceback is not None else error.__traceback__),
        extra={"tara_operation": operation, "tara_error_type": type(error).__name__},
    )
