import torch
from torch import nn
from torchvision import models

# vgg11
vgg11_dogcat = models.vgg11(pretrained=False, num_classes=2)
# vgg11_dogcat.load_state_dict(torch.load('./DogCat/vgg11_best.pth'))
