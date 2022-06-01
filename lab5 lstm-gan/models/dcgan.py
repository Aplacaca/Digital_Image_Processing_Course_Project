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
from .conv_lstm import ConvLSTMCell, ConvLSTM

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        # images of one batch: (40, 1, 256, 256)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False), # (40, 16, 128, 128)
            *discriminator_block(16, 32), # (40, 32, 64, 64)
            *discriminator_block(32, 64), # (40, 64, 32, 32)
            *discriminator_block(64, 128), # (40, 128, 16, 16)
            *discriminator_block(128, 256), # (40, 256, 8, 8)
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 5
        self.adv_layer = nn.Sequential(
            nn.Linear(256 * ds_size ** 2, 1), nn.Sigmoid())

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()

        # images of one batch: (40, 1, 256, 256)
        # shape of input: (40, opt.latent_dim=100)

        self.init_size = opt.img_size // 4 # 64
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128), 
            # nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False), # (40, 128, 128, 128)
            nn.Upsample(scale_factor=2), # (40, 128, 128, 128)
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.ConvTranspose2d(128, 128, 4, 2, padding=1, bias=False), # (40, 64, 256, 256)
            nn.Upsample(scale_factor=2), # (40, 128, 256, 256)
            nn.Conv2d(128, 64, 3, stride=1, padding=1), # (40, 64, 256, 256)
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), # (40, 1, 256, 256)
            nn.Tanh(),
        )

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, z):
        out = self.l1(z)
        # import pdb;pdb.set_trace()
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    
class Conv_Generator(nn.Module):
    def __init__(self, opt):
        super(Conv_Generator, self).__init__()

        # images of one batch: (40, 1, 256, 256)
        # shape of input: (40, opt.latent_dim=100)

        self.init_size = opt.img_size // 4 # 64
        self.bn = nn.BatchNorm2d(128)
        # self.prep = torch.nn.ConvTranspose2d(in_channels, out_channels=3, kernel_size=3)
        self.encoder = ConvLSTM(None,input_channels=1, hidden_channels=[32,64], kernel_size=5, step=1,
                        effective_step=[1])
        self.decoder = ConvLSTM(None,input_channels=64, hidden_channels=[32,64], kernel_size=5, step=1,
                        effective_step=[1])
        self.outprocess = nn.Sequential(
            nn.Conv2d(64, 16, 3, 1,'same'),
            nn.Conv2d(16, 1, 3, 1,'same'),
            # nn.Conv2d(4, 1, 3, 1,'same'),
            nn.Tanh()
        )
        # Initialize weights
        # self.apply(weights_init_normal)

    def forward(self, z):
        out = self.encoder(z)
        out = self.decoder(out[1][0])
        out = self.outprocess(out[1][0])
        # import pdb;pdb.set_trace()
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        return out

class Conv_Generator_1(nn.Module):
    def __init__(self, opt):
        super(Conv_Generator_1, self).__init__()

        # images of one batch: (40, 1, 256, 256)
        # shape of input: (40, opt.latent_dim=100)

        self.init_size = opt.img_size // 4 # 64
        self.bn = nn.BatchNorm2d(128)
        # self.prep = torch.nn.ConvTranspose2d(in_channels, out_channels=3, kernel_size=3)
        self.encoder = ConvLSTM(None,input_channels=1, hidden_channels=[32,64], kernel_size=3, step=1,
                        effective_step=[1])
        self.decoder = ConvLSTM(None,input_channels=64, hidden_channels=[32,16], kernel_size=3, step=1,
                        effective_step=[1])
        self.outprocess = nn.Sequential(
            # nn.Conv2d(64, 16, 3, 1,'same'),
            nn.Conv2d(16, 1, 3, 1,'same'),
            # nn.Conv2d(4, 1, 3, 1,'same'),
            nn.Tanh()
        )
        # Initialize weights
        # self.apply(weights_init_normal)

    def forward(self, z):
        out = self.encoder(z)
        out = self.decoder(out[1][0])
        out = self.outprocess(out[1][0])
        # import pdb;pdb.set_trace()
        # out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        # img = self.conv_blocks(out)
        return out

class Discriminator_1(nn.Module):
    def __init__(self, opt):
        super(Discriminator_1, self).__init__()

        # images of one batch: (40, 1, 256, 256)

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False), # (40, 16, 128, 128)
            *discriminator_block(16, 32), # (40, 32, 64, 64)
            *discriminator_block(32, 64), # (40, 64, 32, 32)
            *discriminator_block(64, 128), # (40, 128, 16, 16)
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity