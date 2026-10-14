"""Finite-difference gradient checking for TARA's autodiff engine.

Reference: Goodfellow, Bengio & Courville, Deep Learning, Chapter 11.
Centered finite differences are used to compare numerical and autodiff gradients.
"""


def numerical_gradient(loss_fn, parameter, epsilon=1e-6):
    """Estimate d(loss_fn)/d(parameter) with a centered finite difference."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    original = parameter.data
    try:
        parameter.data = original + epsilon
        plus = loss_fn()
        plus = float(plus.data if hasattr(plus, "data") else plus)
        parameter.data = original - epsilon
        minus = loss_fn()
        minus = float(minus.data if hasattr(minus, "data") else minus)
        return (plus - minus) / (2.0 * epsilon)
    finally:
        parameter.data = original


def relative_error(analytic, numerical, floor=1e-12):
    """Return a scale-aware absolute relative error."""
    return abs(analytic - numerical) / max(floor, abs(analytic) + abs(numerical))


def check_parameter_gradients(loss_fn, parameters, epsilon=1e-6, tolerance=1e-5, absolute_tolerance=1e-8):
    """Compare autodiff gradients against finite differences.

    Returns a list of diagnostic dictionaries; raises AssertionError on failure.
    """
    diagnostics = []
    for index, parameter in enumerate(parameters):
        numerical = numerical_gradient(loss_fn, parameter, epsilon)
        analytic = parameter.grad
        error = relative_error(analytic, numerical)
        absolute_error = abs(analytic - numerical)
        allowed_error = absolute_tolerance + tolerance * max(abs(analytic), abs(numerical))
        diagnostics.append({
            "index": index,
            "analytic": analytic,
            "numerical": numerical,
            "absolute_error": absolute_error,
            "relative_error": error,
        })
        # Relative error is unstable when both gradients are near zero. Use the
        # standard mixed absolute/relative criterion for a meaningful check.
        if absolute_error > allowed_error:
            raise AssertionError(
                f"gradient check failed at parameter {index}: "
                f"analytic={analytic:.8e}, numerical={numerical:.8e}, "
                f"absolute_error={absolute_error:.8e} > allowed={allowed_error:.8e} "
                f"(relative_error={error:.8e})"
            )
    return diagnostics
