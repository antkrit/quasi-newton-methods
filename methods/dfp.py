"""DFP minimization method."""
from typing import Callable

import numpy as np

from methods.base import Minimizer
from methods.linesearch import linear_search


class DFP(Minimizer):
    """Davidon-Fletcher-Powell Quasi-Newton minimization method.

    This method is a modification of the well-known Newton method in which the first and
    second derivatives are used. Instead of computing Hessian afresh at every iteration,
    we update it in a simple manner to account for the curvature measured during the most
    recent step. Hessian updated with rank 2 matrix.

    Examples:
        Define objective function

        >>> def obj_func(x):
        ...     return x**2

        Initialize minimizer and minimize point x

        >>> minimize = DFP()
        >>> min_ = minimize(np.array([1, -1.5]), obj_func, None, maxiter=100, eps=1e-6)
        >>> np.allclose(min_, [0, 0], atol=1e-6)
        True

    References:
        1. Jorge Nocedal, `Numerical Optimization. Second Edition`, 2006, p.136-139
    """

    def __init__(self):
        self.hessian = None

    def build(self, input_shape):
        self.hessian = np.identity(input_shape[-1])

    def update(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        search_direction = -self.hessian.dot(df(x))
        alpha, *_ = linear_search(x, search_direction, f, df)

        dx = alpha * search_direction  # x_{k+1}

        # convert to 2 by 1 vectors
        dy = np.atleast_2d(df(x + dx) - df(x)).T
        dx = np.atleast_2d(dx).T

        # update Hessian by rank 2 matrix
        np.add(
            self.hessian,
            -self.hessian.dot(dy).dot(dy.T).dot(self.hessian) / dy.T.dot(self.hessian).dot(dy),
            out=self.hessian,
        )
        np.add(self.hessian, dx.dot(dx.T) / dy.T.dot(dx), out=self.hessian)

        return dx.flatten()
