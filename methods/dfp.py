"""DFP minimization method."""
from typing import Callable

import numpy as np

from methods.base import QuasiNewton


class DFP(QuasiNewton):
    """Davidon-Fletcher-Powell minimization method.

    Examples:
        Define objective function

        >>> def obj_func(x):
        ...     return (x**2).sum(axis=0)

        Initialize minimizer and minimize point x

        >>> minimize = DFP()
        >>> min_ = minimize(np.array([1, -1.5]), obj_func, None, maxiter=100, eps=1e-6).x
        >>> np.allclose(min_, [0, 0], atol=1e-6)
        True

    References:
        1. Jorge Nocedal, `Numerical Optimization. Second Edition`, 2006, p.136-139
    """

    def update(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        dx, dy = super().update(x, f, df, *args, **kwargs)

        # update Hessian by rank 2 matrix
        np.add(
            self.hessian,
            -self.hessian.dot(dy).dot(dy.T).dot(self.hessian) / dy.T.dot(self.hessian).dot(dy),
            out=self.hessian,
        )
        np.add(self.hessian, dx.dot(dx.T) / dy.T.dot(dx), out=self.hessian)

        return dx.flatten()
