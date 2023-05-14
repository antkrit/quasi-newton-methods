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
    eps_matrix = _eps_matrix(len(x), eps)
    x = x + np.zeros_like(eps_matrix)  # convert x to the appropriate shape
    return (func(x + eps_matrix) - func(x)) / eps


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
    eps_matrix = _eps_matrix(len(x), eps)
    x = x + np.zeros_like(eps_matrix)  # convert x to the appropriate shape
    return (func(x) - func(x - eps_matrix)) / eps


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
    eps_matrix = _eps_matrix(len(x), eps)
    return (func(x + eps_matrix / 2) - func(x - eps_matrix / 2)) / eps


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

        >>> f = lambda x: np.cos(x).sum(axis=0)
        >>> df = lambda x: -np.sin(x)

        Compare finite difference with exact derivative:

        >>> x = np.array([1, 1])
        >>> np.allclose(finite_difference(x, f), df(x), atol=1e-3)
        True

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

    return _finite_difference(func, np.round(x, 9), eps=eps)
