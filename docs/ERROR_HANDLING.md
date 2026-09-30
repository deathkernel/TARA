# TARA Exception Handling

TARA uses a centralized exception vocabulary in src/error_handling.py.

## Rules

1. Validate early. Use TARAValidationError for invalid runtime state or input.
2. Do not silently swallow failures. If recovery is intentional, log the exception with operation context and continue with a documented fallback.
3. Preserve causes. When an exception is normalized, use exception chaining so the original error remains available.
4. Protect external boundaries. Tool, orchestration, model-provider, filesystem, network, and experiment boundaries should record operation context.
5. Separate failures from hangs. Exception handlers catch raised exceptions; they cannot interrupt code that never raises. Long-running operations therefore need explicit bounds, loop limits, or a process-level watchdog.
6. Never catch KeyboardInterrupt/SystemExit as ordinary application failures.
7. Tests must exercise failure paths, including malformed input, provider failure, retry exhaustion, and timeout/boundary behavior where applicable.

## Exception taxonomy

- TARAError: base class for expected TARA operational failures.
- TARAConfigurationError: invalid or incomplete configuration.
- TARAValidationError: invalid input or violated invariant.
- TARAExecutionError: bounded operation failed.
- TARATimeoutError: explicit operation time budget exceeded.
- TARARetryExhaustedError: all permitted retries failed.
- TARASafetyError: safety or permission boundary rejected an operation.

## Boundary pattern

Use guarded("operation.name") when a public boundary needs consistent logging and normalization. For recoverable fallbacks, call log_exception(...) and return the documented fallback rather than using except Exception: pass.

The goal is not to wrap every line in try/except. Blanket wrapping hides programming defects and makes debugging harder. The strong design is centralized primitives + explicit boundary integration + tests + bounded execution.
