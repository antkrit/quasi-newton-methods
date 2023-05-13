"""Test base classes."""
import numpy as np
import pytest

from methods.base import Minimizer


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_base_minimizer(mocker):
    """Test base minimizer class."""
    # pylint: disable=abstract-class-instantiated
    with pytest.raises(TypeError):
        _ = Minimizer()

    mocker.patch.multiple(Minimizer, __abstractmethods__=set())
    mocker.patch.object(
        Minimizer,
        "update",
        return_value=[-0.5, -0.5],
    )

    def obj_func(x):
        return x

    def obj_grad(x):
        return np.ones_like(x)

    minimizer = Minimizer()
    assert np.array_equal(
        minimizer(np.array([1, 1]), obj_func, obj_grad, maxiter=1).x,
        [0.5, 0.5],
    )
    assert np.array_equal(
        minimizer(np.array([1, 1]), obj_func, obj_grad, maxiter=2).x,
        [0, 0],
    )
    assert np.array_equal(
        minimizer(np.array([1, 1]), obj_func, None, maxiter=2).x,
        [0, 0],
    )
