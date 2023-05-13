"""Base Minimizer class."""
# import annotations to use the literal | between Callable and None
# see arguments of the `Minimizer.__call__()` method
from __future__ import annotations

import abc
import functools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from methods.gradient import finite_difference
from methods.linesearch import linear_search


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

            def update(self, x, f, df, *args, **kwargs):
                return -0.5 * self.some_var * df(x)

    Example:
        >>> import numpy as np

        Create some minimizer

        >>> class SimpleGradientDescent(Minimizer):
        ...     def __init__(self):
        ...         self.some_var = None
        ...     def build(self, input_shape):
        ...         self.some_var = np.ones(input_shape)
        ...     def update(self, x, f, df, *args, **kwargs):
        ...         return -0.5 * self.some_var * df(x)

        Define objective function and its derivative

        >>> def obj_func(x):
        ...     return (x**2).sum(axis=0)
        >>> def obj_grad(x):
        ...     return 2*x

        Initialize minimizer and minimize point x

        >>> x = np.array([0.5, 0.5])
        >>> gradient_descent = SimpleGradientDescent()
        >>> gradient_descent(x, obj_func, obj_grad, maxiter=1).x
        array([0., 0.])

        Or using finite difference

        >>> min_ = gradient_descent(x, obj_func, None, maxiter=1).x
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
        f: Callable[[np.ndarray], np.ndarray],
        df: Callable[[np.ndarray], np.ndarray],
        *args,
        **kwargs,
    ) -> np.ndarray:
        """Update step.

        Arguments:
            x: current x point
            f: target function
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
    ) -> MinimizeResult:
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
            x: result of the minimization
            success: whether the result converged in the allotted number of iterations
            nit: the number of iterations made
        """
        if df is None:
            df = functools.partial(finite_difference, func=f)

        x = np.atleast_1d(x)
        x = x.astype(float)

        self.build(x.shape)

        for i in range(maxiter):
            if np.linalg.norm(df(x)) < eps:
                return MinimizeResult(x, success=True, nit=i)

            np.add(x, self.update(x, f, df, *args, **kwargs), out=x)

        return MinimizeResult(x, success=False, nit=maxiter)


@dataclass
class MinimizeResult:
    """The result of the minimizer."""

    x: np.ndarray
    success: bool
    nit: int


class QuasiNewton(Minimizer):
    """Base quasi-Newton class.

    Quasi-Newton methods is a modification of the well-known Newton method in which the first and
    second derivatives are used. Instead of computing Hessian afresh at every iteration,
    we update it in a simple manner to account for the curvature measured during the most
    recent step.
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
    ) -> tuple:
        """General quasi-Newton update strategy.

        Finds an alpha that satisfies the Wolfe conditions and returns the values of dx and dy

        Note:
            Subclasses must call the parent function in their implementation.

        Arguments:
            x: current x point
            f: target function
            df: target function derivative
            *args: positional arguments to be passed to linear search method
            **kwargs: keyword arguments to be passed to linear search method

        Returns
            dx: n by 1 vector, the difference between x_{k+1} and x_k
            dy: n by 1 vector, the difference between df(x_{k+1}) and df(x_k)
        """
        search_direction = -self.hessian.dot(df(x))

        alpha, *_ = linear_search(x, search_direction, f, df, *args, **kwargs)
        dx = alpha * search_direction  # x_{k+1} - x_{k}

        # convert to 2 by 1 vectors
        dy = np.atleast_2d(df(x + dx) - df(x)).T
        dx = np.atleast_2d(dx).T

        return dx, dy
