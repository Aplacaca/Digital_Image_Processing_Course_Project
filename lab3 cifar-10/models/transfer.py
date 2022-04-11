# -.-coding:utf-8 -.-

from .Basic import BasicModule

from torch.nn import functional as F
from torch import nn
from torchvision import models


class ResNet18_transfer(BasicModule):
    # transfer learning
    def __init__(self, num_classes=10):
        super(ResNet18_transfer, self).__init__()
        self.resnet = models.resnet18(
            pretrained=True, num_classes=1000)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = F.relu(x)
        x = self.fc(x)
        return x