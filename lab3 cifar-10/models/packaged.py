from torch import nn
from torchvision import models

# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 加载预训练好的模型，如果不存在会进行下载
# 预训练好的模型保存在 ~/.torch/models/下面
resnet18 = models.resnet18(pretrained=False, num_classes=10)
resnet34 = models.resnet34(pretrained=False, num_classes=10)
resnet50 = models.resnet50(pretrained=False, num_classes=10)
resnet101 = models.resnet101(pretrained=False, num_classes=10)
resnet152 = models.resnet152(pretrained=False, num_classes=10)
google = models.googlenet(pretrained=False, num_classes=10)

# # 修改最后的全连接层为10分类问题（默认是ImageNet上的1000分类）
# resnet34.fc = nn.Linear(512, 10)
# resnet50.fc = nn.Linear(2048, 10)
# resnet101.fc = nn.Linear(2048, 10)
# resnet152.fc = nn.Linear(2048, 10)
# google.fc = nn.Linear(1024, 10)
