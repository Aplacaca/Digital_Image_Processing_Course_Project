import numpy as np
from .Modules import Module

import torch
import torch.nn.functional as F


class Sigmoid(Module):

    def forward(self, x):
        """Forward propagation of Sigmoid.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        self.x = x

        return 1/(1+np.exp(-x))

    def backward(self, dy):
        """Backward propagation of Sigmoid.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """

        return dy * np.exp(-self.x)/((1+np.exp(-self.x))**2)


class ReLU(Module):

    def forward(self, x):
        """Forward propagation of ReLU.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, L_out).
        """

        self.x = x

        return np.where(x > 0, x, 0)

    def backward(self, dy):
        """Backward propagation of ReLU.

        Args:
            dy: output delta of shape (N, L_out).
        Returns:
            dx: input delta of shape (N, L_in).
        """
        dy[self.x < 0] = 0

        return dy


class argmax(Module):

    def forward(self, x):
        """Forward propagation of ReLU.

        Args:
            x: input of shape (N, L_in).
        Returns:
            out: output of shape (N, 1).
        """
        self.x = x
        self.out = np.argmax(x, axis=1).reshape(-1, 1) - 1

        return self.out

    def backward(self, dy):
        """Backward propagation of Sigmoid.

        Args:
            dy: output delta of shape (N, 1).
        Returns:
            dx: input delta of shape (N, L_in).
        """
        dy = np.zeros_like(self.x)
        for i in range(self.x.shape[0]):
            dy[i][self.out[i]] = 1

        return dy


class Softmax(Module):

    def __init__(self):
        pass

    def forward(self, x):
        """Forward propagation of Softmax.

        Args:
            x: input of shape (batch_size, num_class).
        Returns:
            out: output of shape (batch_size, num_class).
        """

        # self.x_debug = torch.Tensor(x).requires_grad_(True)
        # self.y_debug = F.softmax(self.x_debug, dim=1)

        self.x = x
        max_x = np.max(x, axis=1, keepdims=True)
        x_cen = x - max_x
        exp_x = np.exp(x_cen) + 1e-7
        sum_exp = np.sum(exp_x, axis=1, keepdims=True)
        self.y = exp_x/sum_exp

        # self.x = x
        # x_exp = np.exp(x)
        # self.y = x_exp / x_exp.sum(axis=1, keepdims=True)
        # self.y = np.where(self.y > 1e-45, self.y, 0)

        return self.y

    def backward(self, dy):
        """Backward propagation of Softmax.

        Args:
            dy: output delta of shape (batch_size, num_class).
        Returns:
            dx: input delta of shape (batch_size, num_class).
        """

        # out = np.zeros_like(dy)
        # for j in range(dy.shape[1]):
        #     for i in range(dy.shape[1]):
        #         if i == j:
        #             out[:, j] += self.y[:, j] * (1 - self.y[:, i])
        #         else:
        #             out[:, j] += -self.y[:, j] * self.y[:, i]

        #         if j == 3:
        #             print("[%d][%d]--%.4e*(%d-%.4e)=%.4e--out:%.4e" % (j, i, self.y[99][j],
        #                   (j == i), self.y[99][i], self.y[99][j]*((j == i)-self.y[99][i]), out[99][j]))
        # out = out * 1e16

        # self.x_debug.backward(torch.Tensor(self.y_debug))

        # a1 = np.expand_dims(self.y, -1)
        # a2 = a1.transpose(0, 2, 1)
        # a3 = np.einsum('ijk,ikn->ij', a1, a2)
        # a3 = self.y - a3

        # import ipdb
        # ipdb.set_trace()
        return dy
        # return out * dy


class Loss:
    """
    Usage:
        >>> criterion = Loss(n_classes)
        >>> ...
        >>> for epoch in n_epochs:
        ...     ...
        ...     predict = model(x)
        ...     loss = criterion(probs, target)
        ...     model.backward(loss.backward())
        ...     ...
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __call__(self, predict, targets):
        self.predict = predict
        self.targets = targets
        ...
        return self

    def backward(self):
        ...


class MSELoss(Loss):

    def __call__(self, predict, targets):
        """Forward propagation of MSELoss.

        Args:
            predict: input of shape (1).
            targets: input of shape (1).
        Returns:
            loss: output of shape (1).
        """
        self.x = predict
        self.y = targets
        self.loss = 0.5*np.mean((targets - predict)**2)

        return self

    def backward(self,):
        """
        Backward propagation of MSELoss.
        """

        if len(self.x.shape) < 2:
            self.x = np.expand_dims(self.x, -1)
        if len(self.y.shape) < 2:
            self.y = np.expand_dims(self.y, -1)

        dy = self.x - self.y

        return dy


class CrossEntropy(Loss):

    def __call__(self, predict, targets):
        """Forward propagation of CrossEntropy Loss.
           Must after a softmax.

        Args:
            predict: input of shape (batch_size, num_class).
            targets: input of shape (batch_size, num_class).
        Returns:
            loss: output of shape (1).
        """
        # import pdb;pdb.set_trace()
        self.x = predict

        if targets.shape[-1] == 1:
            targets = targets.squeeze(-1)
        I = np.eye(self.n_classes)
        self.y = I[targets]
        self.delta = 1e-10
        self.loss = np.mean(-np.sum(self.y *
                                    np.log(self.x + 1e-7), axis=1))

        return self

    def backward(self,):
        """
        Backward propagation of CrossEntropyLoss.
        """
        if len(self.x.shape) < 2:
            self.x = np.expand_dims(self.x, -1)
        if len(self.y.shape) < 2:
            self.y = np.expand_dims(self.y, -1)

        # dy = (-self.y / (self.x + self.delta) / self.y.shape[0])
        dy = self.x - self.y
        return dy
