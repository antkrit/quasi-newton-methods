"""Contains functions to compute gradient of the function."""
from typing import Callable

import numpy as np


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
    return (func(x + eps) - func(x)) / eps


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
    return (func(x) - func(x - eps)) / eps


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
    return (func(x + eps / 2) - func(x - eps / 2)) / eps


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

        >>> f = lambda x: np.cos(x)
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
    x = np.atleast_1d(x)

    if x.ndim != 1:
        raise ValueError(f"1d array expected, received: {x.ndim}")

    types = {
        "F": _forward_finite_difference,
        "B": _backward_finite_difference,
        "C": _central_finite_difference,
    }

    _finite_difference = types.get(type_, None)
    if _finite_difference is None:
        raise ValueError(f"Unknown type {type_} available types: {list(types.keys())}")

    return _finite_difference(func, x, eps=eps)
