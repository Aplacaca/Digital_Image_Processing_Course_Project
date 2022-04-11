# -.-coding:utf-8 -.-

from .Basic import BasicModule

from torch.nn import functional as F
from torch import nn


class LrkNet(BasicModule):

    def __init__(self, num_classes=10):
        self.main = nn.Sequential(
            # 3x32x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),  # 3x32x32 (O = (N+2P-F/S)+1)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x16x16
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 32x16x16
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x8x8
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, 3, padding=1),  # 64x8x8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 128x4x4
            nn.BatchNorm2d(128),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        # Conv and Poolilng layers
        x = self.main(x)

        # Flatten before Fully Connected layers
        x = x.view(-1, 128*4*4)

        # Fully Connected Layer
        x = self.fc(x)
        return x
