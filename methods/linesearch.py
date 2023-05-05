"""Contains linear search methods."""
import functools
import warnings
from typing import Callable

import numpy as np

from methods.gradient import finite_difference

# most of the variables in the algorithms have a
# simplified name that should not make any sense
# pylint: disable=invalid-name


def linear_search(
    xk: np.ndarray,
    pk: np.ndarray,
    f: Callable[[np.ndarray, ...], np.ndarray],
    df: Callable[[np.ndarray, ...], np.ndarray] = None,
    c1: float = 0.001,
    c2: float = 0.9,
    amax: float = 2,
    maxiter: int = 10,
    options: dict = None,
    **kwargs,
) -> tuple:
    """Find x that satisfies strong Wolfe conditions.

    Args:
        xk: starting point.
        pk: search direction.
        f: target function
        df (optional): target function derivative. Use finite difference method if omitted.
            Defaults to None.
        c1 (optional): hyperparameter for Armijo condition rule. Defaults to 0.001
        c2 (optional): hyperparameter for curvature condition rule, should be greater than c1.
            Defaults to 0.9.
        amax (optional): maximum a value. Should be greater than 0. Trial point a1
            will be calculated as (0 + amax)/2. Defaults to 2.
        maxiter (optional): maximum number of iterations to perform. Defaults to 100.
        options (optional): additional kwargs passed to target function. Defaults to None
        **kwargs: keyword arguments passed to target function derivative

    Returns:
        alpha: point that satisfies Wolfe conditions.
        fval: new function value f(xk + alpha*pk).
        old_fval: old function value f(xk)

    Example:
        >>> import numpy as np

        Define objective function and it's derivative

        >>> def obj_func(x):
        ...     return (x**2).sum()
        >>> def obj_grad(x):
        ...     return 2*x

        Search for alpha tha satisfy strong Wolfe conditions

        >>> start_point = np.array([1, 1])
        >>> search_gradient = np.array([-1, -1])
        >>> linear_search(start_point, search_gradient, obj_func, obj_grad)
        (0.9990234375, 1.9073486328125e-06, 2)

        It is also possible not to specify the derivative of the objective function, in which
        case the finite difference method will be used. You can pass additional arguments to
        this finite difference function using **kwargs

        >>> linear_search(start_point, search_gradient, obj_func, None)
        (0.9990234375, 1.9073486328125e-06, 2)

    References:
        1. Jorge Nocedal, `Numerical Optimization. Second Edition`, 2006, p.56-62
    """
    options = options or {}
    f = functools.partial(f, **options)

    # if df is not provided, then compute finite
    # difference for the function f
    if df is None:
        df = functools.partial(finite_difference, func=f)

    df = functools.partial(df, **kwargs)

    def phi(alpha):
        """Wrap the f function so that alpha is the only argument passed."""
        return f(xk + alpha * pk)

    def dphi(alpha):
        """Phi function derivative."""
        return df(xk + alpha * pk)

    alpha_k_1 = 0  # previous step
    alpha_k = _interpolate_bisec(alpha_k_1, amax)

    def _output(alpha):
        """Make sure function returns all the values specified in the docstring."""
        return alpha, f(xk + alpha * pk), f(xk)

    for _ in range(1, maxiter):
        phi_ak = phi(alpha_k)
        if np.all(phi_ak > phi(0) + c1 * alpha_k * dphi(0)) or np.all(phi_ak >= phi(alpha_k_1)):
            return _output(_zoom(alpha_k, alpha_k_1, phi, dphi, c1, c2))

        dphi_ak = dphi(alpha_k)
        if np.all(np.abs(dphi_ak) <= -c2 * dphi(0)):
            return _output(alpha_k)

        if np.all(dphi_ak >= 0):
            return _output(_zoom(alpha_k, alpha_k_1, phi, dphi, c1, c2))

        alpha_k_1, alpha_k = alpha_k, _interpolate_bisec(alpha_k, amax)

    warnings.warn("Maximum iteration reached.", RuntimeWarning)
    return _output(alpha_k)


def _zoom(
    xlo: float,
    xhi: float,
    f: Callable[[float], float],
    df: Callable[[float], float],
    c1: float,
    c2: float,
    maxiter: int = 10,
) -> float:
    """Linear search helper.

    See Jorge Nocedal, `Numerical Optimization. Second Edition`, 2006, p.61

    Args:
        xlo: lower bound
        xhi: upper bound
        f: target function
        df: target function derivative.
        c1: hyperparameter for Armijo condition rule. Defaults to 0.001
        c2: hyperparameter for curvature condition rule, should be greater than c1
        maxiter (optional): maximum number of iterations to perform. Defaults to 10.

    Returns:
        Point that satisfies following conditions:

        - the interval bounded by xlo and xhi contains step lengths that satisfy the strong Wolfe
            conditions;
        - xlo is, among all step lengths generated so far and satisfying the sufficient decrease
            condition, the one giving the smallest function value
        - xhi is chosen so that df(xlo)(xhi − xlo) < 0.
    """
    xk = None

    for _ in range(maxiter):
        xk = _interpolate_bisec(xlo, xhi)
        f_xk = f(xk)

        if np.all(f_xk > f(0) + c1 * xk * df(0)) or np.all(f_xk > f(xlo)):
            xhi = xk
        else:
            df_xk = df(xk)
            if np.all(np.abs(df_xk) <= -c2 * df(0)):
                return xk

            if np.all(df_xk * (xhi - xlo) >= 0):
                xhi = xlo

            xlo = xk

    return xk


def _interpolate_bisec(xlo: float, xhi: float) -> float:
    """Bisection method.

    Find the trial minimum of the function in the given range.

    Arguments:
        xlo: lower limit
        xhi: upper limit

    Returns:
         Average of xlo and xhi
    """
    return (xlo + xhi) / 2


def _interpolate_quadratic(
    xlo: float, xhi: float, f: Callable[[float], float], df: Callable[[float], float]
) -> float:
    """Quadratic interpolation method.

    Find minimizer of the function in the given range. See
    `Method of quadratic interpolation`_ for details.

    Arguments:
        xlo: lower limit
        xhi: upper limit
        f: target function
        df: target function derivative

    Returns:
         Minimum of the function between xlo and xhi

    .. _Method of quadratic interpolation:
        https://people.math.sc.edu/kellerlv/Quadratic_Interpolation.pdf
    """
    minimizer = (xhi - xlo) * df(xhi) / (df(xhi) - (f(xhi) - f(xlo) / (xhi - xlo)))
    return xhi - minimizer / 2
