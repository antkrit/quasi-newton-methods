"""Contains utilities to work with arrays."""
import numpy as np


def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=True):  # pylint: disable=invalid-name
    """Numpy `allclose()` wrapper."""
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)
