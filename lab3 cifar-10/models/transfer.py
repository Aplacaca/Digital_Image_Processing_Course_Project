# -.-coding:utf-8 -.-
from torch import nn
from torchvision import models


# resnet18
resnet18_transfer = models.resnet18(pretrained=True)
resnet18_transfer.fc = nn.Linear(resnet18_transfer.fc.in_features, 10)
# resnet18_transfer.fc = nn.Sequential(
#     nn.Linear(resnet18_transfer.fc.in_features, 100, bias=True),
#     nn.ReLU(),
#     nn.Linear(100, 10, bias=True),
# )
# # frozen all layers
# for p in resnet18_transfer.parameters():
#     p.requires_grad = False
# # unfrozen the last layer and fc
# for layer in [resnet18_transfer.layer4.parameters(), resnet18_transfer.fc.parameters()]:
#     for p in layer:
#         p.requires_grad = True
# 查看总参数及训练参数
# total_params = sum(p.numel() for p in resnet18_transfer.parameters())
# print('总参数个数:{}'.format(total_params))
# total_trainable_params = sum(
#     p.numel() for p in resnet18_transfer.parameters() if p.requires_grad)
# print('需训练参数个数:{}'.format(total_trainable_params))


# resnet34
resnet34_transfer = models.resnet34(pretrained=True)
resnet34_transfer.fc = nn.Linear(resnet34_transfer.fc.in_features, 10)


# resnet50
resnet50_transfer = models.resnet50(pretrained=True)
resnet50_transfer.fc = nn.Linear(resnet50_transfer.fc.in_features, 10)


# resnet101
resnet101_transfer = models.resnet101(pretrained=True)
resnet101_transfer.fc = nn.Linear(resnet101_transfer.fc.in_features, 10)


# resnet152
resnet152_transfer = models.resnet152(pretrained=True)
resnet152_transfer.fc = nn.Linear(resnet152_transfer.fc.in_features, 10)


# vgg11
vgg11_transfer = models.vgg11(pretrained=True)
vgg11_transfer.classifier[6] = nn.Linear(
    vgg11_transfer.classifier[6].in_features, 10)


# vgg13
vgg13_transfer = models.vgg13(pretrained=True)
vgg13_transfer.classifier[6] = nn.Linear(
    vgg13_transfer.classifier[6].in_features, 10)


# vgg16
vgg16_transfer = models.vgg16(pretrained=True)
vgg16_transfer.classifier[6] = nn.Linear(
    vgg16_transfer.classifier[6].in_features, 10)


# vgg19
vgg19_transfer = models.vgg19(pretrained=True)
vgg19_transfer.classifier[6] = nn.Linear(
    vgg19_transfer.classifier[6].in_features, 10)


# googlenet
googlenet_transfer = models.googlenet(pretrained=True)
googlenet_transfer.fc = nn.Linear(
    googlenet_transfer.fc.in_features, 10)

# alexnet
alexnet_transfer = models.alexnet(pretrained=True)
alexnet_transfer.classifier[6] = nn.Linear(
    alexnet_transfer.classifier[6].in_features, 10)

# densenet121
densenet121_transfer = models.densenet121(pretrained=True)
densenet121_transfer.classifier = nn.Linear(
    densenet121_transfer.classifier.in_features, 10)

# densenet161
densenet161_transfer = models.densenet161(pretrained=True)
densenet161_transfer.classifier = nn.Linear(
    densenet161_transfer.classifier.in_features, 10)

# densenet169
densenet169_transfer = models.densenet169(pretrained=True)
densenet169_transfer.classifier = nn.Linear(
    densenet169_transfer.classifier.in_features, 10)

# densenet201
densenet201_transfer = models.densenet201(pretrained=True)
densenet201_transfer.classifier = nn.Linear(
    densenet201_transfer.classifier.in_features, 10)