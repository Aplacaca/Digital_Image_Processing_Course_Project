# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan.py
@Time    :   2022/05/22 17:19:08
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Nets for simple DCGAN
"""

import torch
import torch.nn as nn

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# class Discriminator(nn.Module):
#     def __init__(self, opt):
#         super(Discriminator, self).__init__()

#         # images of one batch: (batch_size, 1, 256, 256)

#         def discriminator_block(in_filters, out_filters, bn=True):
#             block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
#                 0.2, inplace=True), nn.Dropout2d(0.25)]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.model = nn.Sequential(
#             *discriminator_block(opt.channels, 16, bn=False), # (batch_size, 16, 128, 128)
#             *discriminator_block(16, 32), # (batch_size, 32, 64, 64)
#             *discriminator_block(32, 64), # (batch_size, 64, 32, 32)
#             *discriminator_block(64, 128), # (batch_size, 128, 16, 16)
#             # *discriminator_block(128, 256), # (batch_size, 256, 8, 8)
#         )

#         # The height and width of downsampled image
#         ds_size = opt.img_size // 2 ** 4
#         self.adv_layer = nn.Sequential(
#             nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

#         # Initialize weights
#         self.apply(weights_init_normal)

#     def forward(self, img):
#         out = self.model(img)
#         out = out.view(out.shape[0], -1)
#         validity = self.adv_layer(out)

#         return validity

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        # images of one batch: (batch_size, 1, 256, 256)
        # shape of input: (batch_size, opt.latent_dim=100)

        self.init_size = opt.img_size // 2 ** 2 # 64
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), 
            # nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False), # (batch_size, 128, 128, 128)
            nn.Upsample(scale_factor=2), # (batch_size, 128, 128, 128)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False), # (batch_size, 64, 256, 256)
            nn.Upsample(scale_factor=2), # (batch_size, 128, 256, 256)
            nn.Conv2d(128, 64, 3, stride=1, padding=1), # (batch_size, 64, 256, 256)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), # (batch_size, 1, 256, 256)
            nn.Tanh(),
        )
        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img