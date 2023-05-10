"""Test minimization methods."""
import numpy as np
import pytest

from methods.dfp import DFP
from tests.utils.array import allclose
from tests.utils.functions import parabola, parabola_derivative


@pytest.mark.parametrize("x", [np.ones(2), np.ones(3)], ids=["2elem.", "3elem."])
def test_dfp(x):
    """Test DFP minimization method."""
    dfp = DFP()

    min_ = dfp(x, parabola, parabola_derivative, eps=1e-6, maxiter=100)
    assert allclose(min_, np.zeros(len(x)), atol=1e-6)

    min_ = dfp(x, parabola, None, eps=1e-6, maxiter=100)
    assert allclose(min_, np.zeros(len(x)), atol=1e-6)
