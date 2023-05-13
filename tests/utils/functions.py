"""Contains implementations of mathematical functions and their derivatives."""
import numpy as np

# pylint: disable=invalid-name


def cos(x):
    """Simple cos function."""
    return np.cos(x).sum(axis=0)


def cos_grad(x):
    """Cos derivative."""
    return -np.sin(x)


def parabola(x):
    """Simple parabolic function."""
    return (x**2).sum(axis=0)


def parabola_grad(x):
    """Parabola derivative."""
    return 2 * x


def rosenbrock(x, a=1, b=100):
    """Rosenbrock function."""
    # 1st row - xs, 2nd - ys
    # see `_finite_difference()` implementation
    x, y = x.T
    return (a - x) ** 2 + b * (y - x**2) ** 2


def rosenbrock_grad(x, a=1, b=100):
    """Rosenbrock derivative."""
    x, y = x
    return np.array([2 * (x - a) - 4 * b * x * (y - x**2), 2 * b * (y - x**2)])
