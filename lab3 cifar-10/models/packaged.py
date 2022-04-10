from torch import nn
from torchvision import models

# 全局取消证书验证
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# 加载预训练好的模型，如果不存在会进行下载
# 预训练好的模型保存在 ~/.torch/models/下面
resnet34 = models.resnet34(pretrained=True, num_classes=1000)
resnet50 = models.resnet50(pretrained=True, num_classes=1000)
resnet101 = models.resnet101(pretrained=True, num_classes=1000)
google = models.googlenet(pretrained=True, num_classes=1000)

# 修改最后的全连接层为10分类问题（默认是ImageNet上的1000分类）
resnet34.fc = nn.Linear(512, 10)
resnet50.fc = nn.Linear(2048, 10)
resnet101.fc = nn.Linear(2048, 10)
google.fc = nn.Linear(1024, 10)
