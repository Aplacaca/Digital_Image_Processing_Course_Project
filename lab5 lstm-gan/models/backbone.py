# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   backbone.py
@Time    :   2022/05/16 10:55:39
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Network backbone to extractor image features
"""

import torch
import torch.nn as nn


def init_layer(m):
    """ 初始化模型参数 """

    classname = m.__class__.__name__
    if isinstance(classname, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(classname, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(classname, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def vgg16(cfg: list, batch_norm=False) -> nn.ModuleList:
    """ 创建 vgg16 模型

    Parameters
    ----------
    cfg : list
        参数列表
    batch_norm: bool
        是否在卷积层后面添加批归一化层
    """
    
    layers = []
    in_channels = 1

    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2, 2))
        elif v == 'C':
            layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
        else:
            conv = nn.Conv2d(in_channels, v, 3, padding=1)

            # 如果需要批归一化的操作就添加一个批归一化层
            if batch_norm:
                layers.extend([conv, nn.BatchNorm2d(v), nn.ReLU(True)])
            else:
                layers.extend([conv, nn.ReLU(True)])

            in_channels = v

    
    layers = nn.ModuleList(layers)
    return layers


class FeatureExtractor(nn.Module):
    """ 创建特征提取器

    Parameters
    ----------
    vgg16_model: nn.ModuleList
        vgg16 模型
    """
    #输入为 (40, 1, 256, 256)，vgg提取后变化为 (N, 32, 16, 16)
    cfg = [16, 16, 'M', 32, 32, 'M', 64, 64,
           64, 'C', 32, 32, 32, 'M']

    def __init__(self, img_size, latent_dim):
        super(FeatureExtractor, self).__init__()
        self.features = vgg16(self.cfg, batch_norm=True)

        num_pool = sum([x.__class__.__name__ == 'MaxPool2d' for x in self.features]) # vgg16中的池化层的数量
        self.fc = nn.Sequential(
            nn.Linear(int(32 * (img_size/2**num_pool)**2), 1024),
            nn.ReLU(True),
            nn.Linear(1024, latent_dim),
            nn.Tanh()
        )
        
        # 初始化模型参数
        for layer in self.features:
            init_layer(layer)
        for layer in self.fc:
            init_layer(layer)

    def forward(self, x):
        """ 前向传播

        Parameters
        ----------
        x: torch.Tensor
            输入图像
        """

        for layer in self.features:
            x = layer(x)
        
        x = x.view(x.size(0), -1)
        outputs = self.fc(x)
            
        return outputs
    