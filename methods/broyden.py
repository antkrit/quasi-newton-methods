"""Broyden minimization method."""
from typing import Callable

import numpy as np

from methods.base import QuasiNewton


class Broyden(QuasiNewton):
    """Broyden minimization method.

    Examples:
        Define objective function

        >>> def obj_func(x):
        ...     return x**2

        Initialize minimizer and minimize point x

        >>> minimize = Broyden()
        >>> min_ = minimize(np.array([1, -1.5]), obj_func, None, maxiter=100, eps=1e-6)
        >>> np.allclose(min_, [0, 0], atol=1e-6)
        True
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

        # update Hessian by rank 1 matrix
        dx_h_dy = (dx - self.hessian.dot(dy)).T
        np.add(
            self.hessian,
            dx_h_dy.T.dot(dx_h_dy) / (dx_h_dy.dot(dy) + 1e-9),
            out=self.hessian,
        )

        return dx.flatten()
