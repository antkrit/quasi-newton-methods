"""Test minimization methods."""
import numpy as np
import pytest

from methods.bfgs import BFGS
from methods.broyden import Broyden
from methods.dfp import DFP
from tests.utils.array import allclose
from tests.utils.functions import parabola, parabola_derivative


@pytest.mark.parametrize("obj_method", [DFP, Broyden, BFGS], ids=["DFP", "Broyden", "BFGS"])
@pytest.mark.parametrize("x", [np.ones(2), np.ones(3)], ids=["2elem.", "3elem."])
def test_convergence(obj_method, x):
    """Test convergence of methods."""
    minimize = obj_method()

    min_ = minimize(x, parabola, parabola_derivative, eps=1e-6, maxiter=100).x
    assert allclose(min_, np.zeros(len(x)), atol=1e-6)

    min_ = minimize(x, parabola, None, eps=1e-6, maxiter=100).x
    assert allclose(min_, np.zeros(len(x)), atol=1e-6)
