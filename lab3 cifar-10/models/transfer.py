# -.-coding:utf-8 -.-

from .Basic import BasicModule

from torch import nn
from torchvision import models


resnet18_transfer = models.resnet18(pretrained=False, num_classes=10)
resnet18_transfer.fc = nn.Linear(resnet18_transfer.fc.in_features, 10)
