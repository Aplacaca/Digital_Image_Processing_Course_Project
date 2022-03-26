from .Modules import Module


class Sigmoid(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of sigmoid function.

        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of sigmoid function.

        ...

        # End of todo


class ReLU(Module):

    def forward(self, x):

        # TODO Implement forward propogation
        # of ReLU function.

        ...

        # End of todo

    def backward(self, dy):

        # TODO Implement backward propogation
        # of ReLU function.

        ...

        # End of todo


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

        # TODO Calculate MSE loss.

        ...

        # End of todo

    def backward(self):

        # TODO Implement backward propogation
        # of mean square error loss function.

        ...

        # End of todo
