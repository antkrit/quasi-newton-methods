"""Test linear search methods."""
import numpy as np
import pytest

from methods.linesearch import linear_search
from tests.utils.functions import parabola, parabola_derivative


@pytest.mark.parametrize(
    "x_k, p_k",
    [
        (1, -2),
        (np.array([-1]), np.array([1])),
        (np.array([1, 1]), np.array([-1, -1])),
    ],
)
def test_linesearch_using_derivative(x_k, p_k):
    """Test function output, if a derivative function is provided."""
    _, fval, old_fval = linear_search(x_k, p_k, parabola, parabola_derivative)
    assert np.all(fval < old_fval)


@pytest.mark.parametrize(
    "x_k, p_k",
    [
        (2, -1),
        (np.array([-1]), np.array([1])),
        (np.array([1, 1]), np.array([-1, -1])),
    ],
)
def test_linesearch_without_derivative(x_k, p_k):
    """Test function output, if a derivative function is omitted."""
    _, fval, old_fval = linear_search(x_k, p_k, parabola, None)
    assert np.all(fval < old_fval)
