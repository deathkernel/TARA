"""Central error-handling primitives for TARA.

This module provides one vocabulary for operational failures, structured
exception records, logging, and boundary handling. It deliberately does not
silently swallow failures. A genuinely hung Python thread needs a bounded
loop or process-level watchdog; an exception handler cannot interrupt code
that never raises.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import logging
import traceback
from typing import Any, Callable, TypeVar

T = TypeVar("T")
LOGGER = logging.getLogger("tara")


class TARAError(Exception):
    """Base class for expected TARA operational failures."""


class TARAConfigurationError(TARAError):
    """Invalid or incomplete runtime configuration."""


class TARAValidationError(TARAError, ValueError):
    """Input or invariant validation failed."""


class TARAExecutionError(TARAError):
    """A bounded operation failed during execution."""


class TARATimeoutError(TARAExecutionError, TimeoutError):
    """An operation exceeded its explicit time budget."""


class TARARetryExhaustedError(TARAExecutionError):
    """A retryable operation failed after all allowed attempts."""


class TARASafetyError(TARAError):
    """A safety or permission boundary rejected an operation."""


@dataclass(frozen=True)
class ExceptionRecord:
    error_type: str
    message: str
    operation: str
    retryable: bool = False
    traceback_text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "message": self.message,
            "operation": self.operation,
            "retryable": self.retryable,
            "traceback": self.traceback_text,
        }


def exception_record(
    exc: BaseException,
    *,
    operation: str = "unknown",
    retryable: bool = False,
    include_traceback: bool = False,
) -> ExceptionRecord:
    """Convert an exception into a stable, serializable diagnostic record."""
    return ExceptionRecord(
        error_type=type(exc).__name__,
        message=str(exc),
        operation=operation,
        retryable=retryable,
        traceback_text=traceback.format_exc() if include_traceback else None,
    )


def log_exception(
    exc: BaseException,
    *,
    operation: str = "unknown",
    retryable: bool = False,
    level: int = logging.ERROR,
) -> ExceptionRecord:
    """Log an exception with context and return its structured record."""
    record = exception_record(
        exc,
        operation=operation,
        retryable=retryable,
        include_traceback=True,
    )
    LOGGER.log(
        level,
        "TARA operation failed: operation=%s type=%s retryable=%s message=%s",
        record.operation,
        record.error_type,
        record.retryable,
        record.message,
        exc_info=True,
    )
    return record


def guarded(
    operation: str,
    *,
    translate: bool = True,
    retryable: Callable[[BaseException], bool] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorate a boundary so failures are logged and optionally normalized."""
    if not operation.strip():
        raise ValueError("operation must not be empty")

    def decorator(function: Callable[..., T]) -> Callable[..., T]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return function(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as exc:
                is_retryable = bool(retryable(exc)) if retryable else False
                log_exception(exc, operation=operation, retryable=is_retryable)
                if not translate or isinstance(exc, TARAError):
                    raise
                raise TARAExecutionError(
                    f"{operation} failed: {type(exc).__name__}: {exc}"
                ) from exc
        return wrapper
    return decorator


def require(
    condition: bool,
    message: str,
    *,
    error_type: type[Exception] = TARAValidationError,
) -> None:
    """Raise a typed TARA validation error when an invariant is false."""
    if not condition:
        raise error_type(message)


__all__ = [
    "ExceptionRecord",
    "LOGGER",
    "TARAError",
    "TARAConfigurationError",
    "TARAExecutionError",
    "TARAValidationError",
    "TARATimeoutError",
    "TARARetryExhaustedError",
    "TARASafetyError",
    "exception_record",
    "log_exception",
    "guarded",
    "require",
]
