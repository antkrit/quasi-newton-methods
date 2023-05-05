"""Contains special warnings."""
import warnings


def invoke_max_iter_warning():
    """Raise a warning when the maximum iteration value is exceeded."""
    warnings.warn("Maximum iteration reached.", RuntimeWarning)


warnings_ = {
    "max-iter": invoke_max_iter_warning,
}
