"""Base Minimizer class."""
# import annotations to use the literal | between Callable and None
# see arguments of the `Minimizer.__call__()` method
from __future__ import annotations

import abc
import functools
from typing import Callable

import numpy as np

from methods.gradient import finite_difference
from methods.warnings import warnings_


class Minimizer(abc.ABC):
    """Base minimizer class.

    To implement your own minimizer, you need to define:

    - `self.update()` method to return the value to which the current x needs to be updated.
    - `self.build()` method to initialize the internal variables of the minimizer at runtime
        based on the size of the input.

    Simple minimizer can be implemented as this::

        class GradientDescent(Minimizer):

            def __init__(self):
                self.some_var = None

            def build(self, input_shape):
                self.some_var = np.ones(input_shape)

            def update(self, x, df, *args, **kwargs):
                return -0.5 * self.some_var * df(x)

    Example:
        >>> import numpy as np

        Create some minimizer

        >>> class SimpleGradientDescent(Minimizer):
        ...     def __init__(self):
        ...         self.some_var = None
        ...     def build(self, input_shape):
        ...         self.some_var = np.ones(input_shape)
        ...     def update(self, x, df, *args, **kwargs):
        ...         return -0.5 * self.some_var * df(x)

        Define objective function and its derivative

        >>> def obj_func(x):
        ...     return x**2
        >>> def obj_grad(x):
        ...     return 2*x

        Initialize minimizer and minimize point x

        >>> x = np.array([0.5, 0.5])
        >>> gradient_descent = SimpleGradientDescent()
        >>> gradient_descent(x, obj_func, obj_grad, maxiter=1)
        array([0., 0.])

        Or using finite difference

        >>> min_ = gradient_descent(x, obj_func, None, maxiter=1)
        >>> np.allclose(min_, [0, 0])
        True
    """

    def build(self, input_shape: tuple) -> None:
        """Initialize the internal variables."""
        del input_shape  # not used in the base class implementation

    @abc.abstractmethod
    def update(
        self,
        x: np.ndarray,
        df: Callable[[np.ndarray], np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Update step.

        Arguments:
            x: current x point
            df: target function derivative

        Returns:
            The value to update x to
        """
        raise NotImplementedError("Must be implemented in subclasses.")

    def __call__(
        self,
        x: np.ndarray,
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray] | None,
        *args,
        eps: float = 1e-8,
        maxiter: int = 100,
        **kwargs,
    ) -> np.ndarray:
        """Minimize x.

        This method simply runs iterations to update x. The update logic should
        be described in the `self.update()` method.

        Arguments:
            x: starting point
            f: target function
            df: target function derivative. Use finite difference method if None.
            eps (optional): convergence tolerance. Defaults to 1e-8
            maxiter (optional): maximum number of iterations. Defaults to 100
            *args: positional arguments passed to update function
            **kwargs: keyword arguments passed to update function

        Returns:
            Minimum of the target function.
        """
        if df is None:
            df = functools.partial(finite_difference, func=f)

        x = np.atleast_1d(x)
        x = x.astype(float)

        self.build(x.shape)

        for _ in range(maxiter):
            if np.linalg.norm(df(x)) < eps:
                return x

            np.add(x, self.update(x, df, *args, **kwargs), out=x)

        warnings_["max-iter"]()
        return x
