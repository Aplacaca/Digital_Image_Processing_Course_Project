# -.-coding:utf-8 -.-

from .Basic import BasicModule

from torch.nn import functional as F
from torch import nn


class LrkNet(BasicModule):

    def __init__(self, num_classes=10):
        super(LrkNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64*8*8, num_classes)

        self.norm = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
