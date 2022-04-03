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

        if self not in graph.dict.values():
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
        for op_idx in graph.dict.keys():
            x = graph.dict[op_idx].forward(x)

    def backward(self, dy: np.ndarray) -> np.ndarray:
        """Defines the backward propagation of the module.
        """
        # import pdb;pdb.set_trace()
        op_rev_list = list(graph.dict.keys())[::-1]
        for op_idx in op_rev_list:
            # print("in "+op_idx+" backward")
            dy = graph.dict[op_idx].backward(dy)
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
        out = x.dot(self.w[1:]) + self.w[0]
        return out

    def backward(self, dy):
        """Backward propagation of linear module.

        Args:
            dy: output delta of shape (L_out).
        Returns:
            dx: input delta of shape (L_in).
        """

        # if dy.shape == None:
        self.w.grad[1:, :] = (self.x.T).dot(dy)
        # else:
        #     self.w.grad[1:, :] = self.x.T.reshape(-1, 1) * dy
        # self.w.grad[0] = dy * 1
        self.w.grad[0] = dy.sum(axis=0) * 1

        return dy.dot(self.w[1:].T)
