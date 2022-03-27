import numpy as np
from .my_tensor import Tensor


class Optim(object):

    def __init__(self, lr):
        self.lr = lr

    def step(self, module):
        self._step_module(module)

    def _step_module(self, module):
        # Traverse the attributes of `self.module`,
        # if is `Tensor`, call `self._update_weight()`,
        # else if is `Module` or `List` of `Module`,
        # call `self._step_module()` recursively.

        if type(module) == Tensor:
            self._update_weight(module)
        elif type(module) == list:
            [self._step_module(module[i]) for i in range(len(module))]

    def _update_weight(self, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, lr, momentum: float = 0):
        super(SGD, self).__init__(lr)
        self.momentum = momentum

    def _update_weight(self, tensor):

        if np.random.random() < 0.8:  # random update
            v = self.momentum * tensor + self.lr * tensor.grad
            # v = self.lr * tensor.grad
            tensor -= v