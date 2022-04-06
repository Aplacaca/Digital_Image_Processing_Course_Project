import numpy as np
from copy import deepcopy
from .my_tensor import Tensor, zeros


class Optim(object):

    def __init__(self, module_params: list, lr: float = 1e-3):
        self.lr = lr
        self.params = module_params

    def step(self):
        for i, param in enumerate(self.params):
            if type(param) != Tensor:
                print('Error: `param` must be `Tensor`.')
                return
            self._update_weight(i, param)

    def _update_weight(self, i, tensor):
        tensor -= self.lr * tensor.grad


class SGD(Optim):

    def __init__(self, module_params: list, lr: float = 1e-4, momentum: float = 0, dampening: float = 0, nesterov: bool = False):
        super(SGD, self).__init__(lr)
        self.params = module_params
        self.momentum = momentum  # Momentum
        self.dampening = dampening
        self.nesterov = nesterov  # NAG

    def _update_weight(self, i, tensor):
        # batch_num = tensor.shape[0]
        # idx = np.random.choice(batch_num,1)
        # target_batch = tensor[0,]
        if self.momentum > 0:
            if tensor.momentum_grad is None:
                tensor.momentum_grad = deepcopy(self.grad)
            else:
                tensor.momentum_grad = self.momentum * \
                    tensor.momentum_grad + (1-self.dampening) * tensor.grad

            if self.nesterov:
                tensor.grad = tensor.grad + self.momentum * tensor.momentum_grad
            else:
                tensor.grad = tensor.momentum_grad

        v = self.lr * tensor.grad
        tensor -= v

        tensor.grad = zeros(tensor.grad.shape)


class Adagrad(Optim):

    def __init__(self, module_params: list, lr: float = 1e-2, lr_decay: float = 0, eps: float = 1e-10):
        super(Adagrad, self).__init__(module_params)
        self.lr = lr
        self.lr_decay = lr_decay
        self.eps = eps
        self.params = module_params

        # statistics
        self.grad_square_sum = [np.zeros_like(self.params[i])
                                for i in range(len(self.params))]
        self.steps = [0 for _ in range(len(self.params))]

    def _update_weight(self, i, tensor):
        self.grad_square_sum[i] += tensor.grad ** 2
        self.steps[i] += 1

        clr = self.lr * (1 / (1 + self.lr_decay * (self.steps[i]-1)))
        v = clr * tensor.grad / (np.sqrt(self.grad_square_sum[i]) + self.eps)
        tensor -= v

        tensor.grad = zeros(tensor.grad.shape)


class RMSProp(Optim):

    def __init__(self, module_params: list, lr: float = 1e-3, alpha: float = 0.99, eps: float = 1e-8, momentum: float = 0):
        super(RMSProp, self).__init__(lr)
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.params = module_params

        # statistics
        self.grad_square_avg = [np.zeros_like(
            self.params[i]) for i in range(len(self.params))]  # Exponential Moving Average
        self.steps = [0 for _ in range(len(self.params))]

    def _update_weight(self, i, tensor):
        self.grad_square_avg[i] = self.alpha * \
            self.grad_square_avg[i] + (1-self.alpha)*tensor.grad ** 2
        self.steps[i] += 1

        if self.momentum > 0:
            if tensor.momentum_grad is None:
                tensor.momentum_grad = deepcopy(self.grad)
            else:
                tensor.momentum_grad = self.momentum * \
                    tensor.momentum_grad + tensor.grad / \
                    self.grad_square_avg[i]
                tensor -= self.lr * tensor.momentum_grad

        v = self.lr * tensor.grad / \
            (np.sqrt(self.grad_square_avg[i]) + self.eps)
        tensor -= v

        tensor.grad = zeros(tensor.grad.shape)


class Adam(Optim):

    def __init__(self, module_params: list, lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0, amsgrad: bool = False):
        super(Adam, self).__init__(module_params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.params = module_params

        # statistics
        self.m = [np.zeros_like(
            self.params[i]) for i in range(len(self.params))]
        self.m_ = [np.zeros_like(self.params[i])
                   for i in range(len(self.params))]
        self.v = [np.zeros_like(
            self.params[i]) for i in range(len(self.params))]
        self.v_ = [np.zeros_like(self.params[i])
                   for i in range(len(self.params))]
        self.steps = [0 for _ in range(len(self.params))]

    def _update_weight(self, i, tensor):
        if self.weight_decay != 0:
            tensor.grad *= self.weight_decay

        self.m[i] = self.betas[0] * self.m[i] + (1-self.betas[0]) * tensor.grad
        self.v[i] = self.betas[1] * self.v[i] + \
            (1-self.betas[1]) * tensor.grad**2
        self.m_[i] = self.m[i] / (1-self.betas[0]**(self.steps[i]+1))
        self.v_[i] = self.v[i] / (1-self.betas[1]**(self.steps[i]+1))

        v = self.lr * self.m_[i] / (np.sqrt(self.v_[i]) + self.eps)
        tensor -= v

        tensor.grad = zeros(tensor.grad.shape)
