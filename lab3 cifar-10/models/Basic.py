# -.-coding:utf-8 -.-
import time
import torch
from torch.nn import Module


class BasicModule(Module):
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def load(self, path):
        """
        load the model
        """

        self.load_state_dict(torch.load(path))

    def save(self, name):
        """
        save the model, named by time
        """

        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        torch.save(self.state_dict(), name)
        return name
