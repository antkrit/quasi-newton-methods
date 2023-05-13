"""Contains special warnings."""
import warnings


def invoke_max_iter_warning(method_name):
    """Raise a warning when the maximum iteration value is exceeded."""
    warnings.warn(f"maximum iteration reached ({method_name}).", RuntimeWarning)


warnings_ = {
    "max-iter": invoke_max_iter_warning,
}
