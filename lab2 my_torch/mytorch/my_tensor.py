import numpy as np
from copy import deepcopy


class Tensor(np.ndarray):
    """Derived Class of np.ndarray."""

    def __init__(self, *args, **kwargs):
        self.grad = None
        self.momentum_grad = None


def tensor(shape):
    """Return a tensor with a normal Gaussian distribution."""
    return random(shape)


def from_array(arr):
    """Convert the input array-like to a tensor."""
    t = arr.view(Tensor)
    t.grad = deepcopy(t)
    t.grad.fill(1)
    t.momentum_grad = deepcopy(t)
    t.momentum_grad.fill(1)
    return t


def zeros(shape):
    """Return a new tensor of given shape, filled with zeros."""
    t = ones(shape)
    t.fill(0)
    return t


def ones(shape):
    """Return a new tensor of given shape, filled with ones."""
    t = tensor(shape)
    t.fill(1)
    return t


def ones_like(tensor):
    """Return a new tensor with the same shape as the given tensor, 
       filled with ones."""
    return ones(tensor.shape)


def random(shape, loc=0.0, scale=1):
    """Return a new tensor of given shape, from normal distribution."""
    return from_array(np.random.normal(loc=loc, scale=scale, size=shape))
