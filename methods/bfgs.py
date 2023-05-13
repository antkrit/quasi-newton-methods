"""BFGS minimization method."""
from typing import Callable

import numpy as np

from methods.base import QuasiNewton


class BFGS(QuasiNewton):
    """Broyden-Fletcher-Goldfarb-Shanno minimization method.

    Examples:
        Define objective function

        >>> def obj_func(x):
        ...     return (x**2).sum(axis=0)

        Initialize minimizer and minimize point x

        >>> minimize = BFGS()
        >>> min_ = minimize(np.array([1, -1.5]), obj_func, None, maxiter=100, eps=1e-6).x
        >>> np.allclose(min_, [0, 0], atol=1e-6)
        True

    References:
        1. Jorge Nocedal, `Numerical Optimization. Second Edition`, 2006, p.140
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

        identity = np.identity(len(x))

        np.dot(
            identity - (dx.dot(dy.T) / dy.T.dot(dx)),
            self.hessian,
            out=self.hessian,
        )
        np.dot(
            self.hessian,
            identity - (dy.dot(dx.T) / dy.T.dot(dx)),
            out=self.hessian,
        )
        np.add(self.hessian, dx.dot(dx.T) / dy.T.dot(dx), out=self.hessian)

        return dx.flatten()
