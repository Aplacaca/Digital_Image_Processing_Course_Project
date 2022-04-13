# torchvision 封装好的模型
from .packaged import resnet18, resnet34, resnet50, resnet101, resnet152, vgg11, vgg13, vgg16, vgg19, dense121, dense161, dense169, dense201

# # 用torchvision预训练的模型迁移学习
from .transfer import googlenet_transfer
#     resnet18_transfer, resnet34_transfer, resnet50_transfer, resnet101_transfer, resnet152_transfer, \
#     vgg11_transfer, vgg13_transfer, vgg16_transfer, vgg19_transfer, \
#     densenet121_transfer, densenet161_transfer, densenet169_transfer, densenet201_transfer, \
#     googlenet_transfer, alexnet_transfer

# 自定义网络结构
from .nets import LrkNet

# # DogCat enhance transfer
from .dogcat_transfer import vgg11_dogcat