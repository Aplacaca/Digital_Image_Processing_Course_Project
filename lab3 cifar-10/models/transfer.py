# -.-coding:utf-8 -.-

from .Basic import BasicModule

from torch import nn
from torchvision import models


# resnet18 from torchvision
resnet18_transfer = models.resnet18(pretrained=True, num_classes=1000)
# resnet18_transfer.fc = nn.Linear(resnet18_transfer.fc.in_features, 10)
resnet18_transfer.fc = nn.Sequential(
    nn.Linear(resnet18_transfer.fc.in_features, 100, bias=True),
    nn.ReLU(),
    nn.Linear(100, 10, bias=True),
)
# frozen all layers
for p in resnet18_transfer.parameters():
    p.requires_grad = False
# unfrozen the last layer and fc
for layer in [resnet18_transfer.layer4.parameters(), resnet18_transfer.fc.parameters()]:
    for p in layer:
        p.requires_grad = True


# resnet50 from torchvision
resnet50_transfer = models.resnet50(pretrained=True, num_classes=1000)
resnet50_transfer.fc = nn.Linear(resnet50_transfer.fc.in_features, 10)
# frozen all layers
for p in resnet50_transfer.parameters():
    p.requires_grad = False
# unfrozen the last layer and fc
for layer in [resnet50_transfer.layer4.parameters(), resnet50_transfer.fc.parameters()]:
    for p in layer:
        p.requires_grad = True
