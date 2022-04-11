# torchvision 封装好的模型
from .packaged import resnet18, resnet34, resnet50, resnet101, resnet152, google

# 用torchvision预训练的模型迁移学习
from .transfer import ResNet18_transfer

# 自定义网络结构
from .nets import ResNet34, LrkNet
