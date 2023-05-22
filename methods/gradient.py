"""Contains functions to compute gradient of the function."""
from typing import Callable

import numpy as np


def _eps_matrix(size, eps):
    """Generate epsilon matrix to match x shape."""
    identity = np.eye(size)
    np.fill_diagonal(identity, np.tile(eps, size))

    return identity


def _forward_finite_difference(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Forward finite difference.

    Args:
        func: target function
        x: values vector
        eps (optional): precision. Defaults to 1e-8

    Returns:
        Approximate gradient
    """
    xeps = x + _eps_matrix(len(x), eps)
    return np.fromiter(((func(xe) - func(x)) / eps for xe in xeps), dtype=float)


def _backward_finite_difference(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Backward finite difference.

    Args:
        func: target function
        x: values vector
        eps (optional): precision. Defaults to 1e-8

    Returns:
        Approximate gradient
    """
    xeps = x - _eps_matrix(len(x), eps)
    return np.fromiter(((func(x) - func(xe)) / eps for xe in xeps), dtype=float)


def _central_finite_difference(
    func: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Central finite difference.

    Args:
        func: target function
        x: values vector
        eps (optional): precision. Defaults to 1e-8

    Returns:
        Approximate gradient
    """
    eps_matrix = _eps_matrix(len(x), eps) / 2
    return np.fromiter(((func(x + e) - func(x - e)) / eps for e in eps_matrix), dtype=float)


FD_TYPES = {
    "F": _forward_finite_difference,
    "B": _backward_finite_difference,
    "C": _central_finite_difference,
}


def finite_difference(
    x: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    eps: float = 1e-8,
    type_: str = "F",
) -> np.ndarray:
    """Calculate gradient approximation using finite difference method.

    Example:
        >>> import numpy as np

        Define objective function and its derivative

        >>> f = lambda x: (x**2).sum(axis=0)
        >>> df = lambda x: 2*x

        Compare finite difference with exact derivative:

        >>> x = np.array([1, 1])
        >>> finite_difference(x, f, eps=1e-6, type_='C')
        array([2., 2.])

    Args:
        x: 1d values vector
        func: target function
        eps (optional): precision. Defaults to 1e-8
        type_ (optional): one of ['F', 'B', 'C'], where 'F' means forward finite difference,
            'B' - backward finite difference, 'C' - central finite difference. Defaults to 'F'.

    Returns:
        Approximate gradient

    Raises:
        ValueError: vector x is not one-dimensional
        ValueError: received unexpected type_ argument
    """
    x = np.atleast_1d(x)  # convert scalars to 1d array

    if x.ndim != 1:
        raise ValueError(f"1d array expected, received: {x.ndim}")

    _finite_difference = FD_TYPES.get(type_, None)
    if _finite_difference is None:
        raise ValueError(f"Unknown type {type_} available types: {list(FD_TYPES.keys())}")

    return _finite_difference(func, x, eps=eps)
