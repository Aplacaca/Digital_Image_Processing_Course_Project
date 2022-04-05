from typing import OrderedDict
from matplotlib.pyplot import axis
import numpy as np
from . import my_tensor
from mytorch.myglobal import graph


class Module(object):
    """Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""

        self.training = True

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """

        idx = len(list(graph.dict.keys()))
        graph.dict[str(idx)] = self

        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        ...

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        ...

    def get_name(self) -> str:
        name = self.__class__.__name__
        return name


class Model(Module):
    """Base class for all neural network modules.
    """

    def __init__(self) -> None:
        """If a module behaves different between training and testing,
        its init method should inherit from this one."""
        self.training = True
        # self.ops = OrderedDict()

    def __call__(self, x: np.ndarray):
        """Defines calling forward method at every call.
        Should not be overridden by subclasses.
        """
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Defines the forward propagation of the module performed at every call.
        Should be overridden by all subclasses.
        """
        graph.flush()
        

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """

        op_rev_list = list(graph.dict.keys())[::-1]
        for op_idx in op_rev_list:
            # print(f"backward in {op_idx} : {graph.dict[op_idx]}")
            dy = graph.dict[op_idx].backward(dy)

        graph.flush()  # 每次backward完成后，清空graph.dict

        return dy

    def get_name(self) -> str:
        name = self.__class__.__name__
        return name


class Linear(Module):

    def __init__(self, in_features: int, out_features: int):
        """Module which applies linear transformation to input.

        Args:
            in_features: L_in from expected input shape (N, L_in).
            out_features: L_out from output shape (N, L_out).
        """

        # w[0] for bias and w[1:] for weight
        self.w = my_tensor.tensor((in_features + 1, out_features))

    def forward(self, x):
        """Forward propagation of linear module.

        Args:
            x: input of shape (in_features, ).
        Returns:
            out: output of shape (out_features, ).
        """
        self.x = x
        if len(self.x.shape) < 2:
            self.x = np.broadcast_to(self.x, [1, self.x.shape[0]])
        out = self.x.dot(self.w[1:]) + self.w[0]
        return out

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (L_out).
        Returns:
            dx: input delta of shape (L_in).
        """

        if len(self.x.shape) < 2:
            self.x = np.broadcast_to(
                self.x, [1, self.x.shape[0]])  # (b,n_feature)
        batch = self.x.shape[0]
        if dy.shape is None:
            dy = np.broadcast_to(dy, [batch, self.w.shape[-1]])
        elif len(dy.shape) < 2:
            dy = np.reshape(dy, (-1, self.w.shape[-1]))

        self.w.grad[1:, :] = (self.x.T).dot(dy)
        self.w.grad[0] = dy.sum(axis=0) * 1

        return dy.dot(self.w[1:].T)
