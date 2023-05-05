"""Test gradient calculations."""
import numpy as np
import pytest

from methods.gradient import finite_difference
from tests.utils.array import allclose
from tests.utils.functions import cos, cos_derivative, sigmoid, sigmoid_derivative


@pytest.mark.parametrize("type_", ["F", "B", "C", "Invalid"])
@pytest.mark.parametrize("eps", [1e-3, 1e-5, 1e-8])
@pytest.mark.parametrize(
    "f, df", [(sigmoid, sigmoid_derivative), (cos, cos_derivative)], ids=("sigmoid", "cos")
)
def test_finite_difference(f, df, eps, type_):
    """Test finite difference function with different parameters."""
    x = np.random.randint(0, 10, size=(5,))

    if type_ == "Invalid":
        with pytest.raises(ValueError):
            _ = finite_difference(x, f, eps=eps, type_=type_)
        return

    derivative = finite_difference(x, f, eps=eps, type_=type_)

    assert allclose(derivative, df(x), atol=1e-3)

    x = np.random.randint(0, 10, size=())
    derivative = finite_difference(x, f, eps=eps, type_=type_)

    assert allclose(derivative, df(x), atol=1e-3)


def test_finite_difference_with_different_shapes():
    """Test function behavior for different x sizes."""

    def f(x):
        """Objective function."""
        return x

    with pytest.raises(ValueError):
        x_3d = np.zeros((2, 2, 2))
        _ = finite_difference(x_3d, f)

    with pytest.raises(ValueError):
        x_2d = np.zeros((2, 2))
        _ = finite_difference(x_2d, f)
