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
            _ = finite_difference(f, x, eps=eps, type_=type_)
        return

    with pytest.raises(ValueError):
        x_2d = np.zeros((2, 2))
        _ = finite_difference(f, x_2d, eps=eps, type_=type_)

    with pytest.raises(ValueError):
        x_2d = np.zeros(())
        _ = finite_difference(f, x_2d, eps=eps, type_=type_)

    derivative = finite_difference(f, x, eps=eps, type_=type_)

    assert allclose(derivative, df(x), atol=1e-3)
