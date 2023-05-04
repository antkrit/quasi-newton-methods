"""Contains implementations of mathematical functions and their derivatives."""
import numpy as np


def sigmoid(x):
    """Simple sigmoid function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """Sigmoid derivative."""
    return sigmoid(x) * (1 - sigmoid(x))


def cos(x):
    """Simple cos function."""
    return np.cos(x)


def cos_derivative(x):
    """Cos derivative."""
    return -np.sin(x)
