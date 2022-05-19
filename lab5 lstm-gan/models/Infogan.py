# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   Infogan.py
@Time    :   2022/05/18 08:29:08
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Nets for InfoGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
	def __init__(self):
		super().__init__()

		# input shape (batch_size, 100+10*10+2, 1, 1)

		self.tconv1 = nn.ConvTranspose2d(100+10*10+2, 512, 2, 1, bias=False) # shape (batch_size, 448, 2, 2)
		self.bn1 = nn.BatchNorm2d(512)
		self.tconv2 = nn.ConvTranspose2d(512, 256, 4, 2, padding=1, bias=False) # shape (batch_size, 256, 4, 4)
		self.bn2 = nn.BatchNorm2d(256)
		self.tconv3 = nn.ConvTranspose2d(256, 128, 4, 2, padding=1, bias=False) # shape (batch_size, 128, 8, 8)
		self.bn3 = nn.BatchNorm2d(128)
		self.tconv4 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False) # shape (batch_size, 64, 16, 16)
		self.bn4 = nn.BatchNorm2d(64)
		self.tconv5 = nn.ConvTranspose2d(64, 32, 4, 2, padding=1, bias=False) # shape (batch_size, 1, 32, 32)
		self.bn5 = nn.BatchNorm2d(32)
		self.tconv6 = nn.ConvTranspose2d(32, 16, 4, 2, padding=1, bias=False) # shape (batch_size, 1, 64, 64)
		self.tconv7 = nn.ConvTranspose2d(16, 8, 4, 2, padding=1, bias=False) # shape (batch_size, 1, 128, 128)
		self.tconv8 = nn.ConvTranspose2d(8, 1, 4, 2, padding=1, bias=False) # shape (batch_size, 1, 256, 256)

		self.apply(weights_init)

	def forward(self, x):
		x = F.relu(self.bn1(self.tconv1(x)))
		x = F.relu(self.bn2(self.tconv2(x)))
		x = F.relu(self.bn3(self.tconv3(x)))
		x = F.relu(self.bn4(self.tconv4(x)))
		x = F.relu(self.bn5(self.tconv5(x)))
		x = F.relu(self.tconv6(x))
		x = F.relu(self.tconv7(x))

		img = torch.tanh(self.tconv8(x))

		return img

class Discriminator(nn.Module):
	def __init__(self):
		super().__init__()

		# input shape (batch_size, 1, 256, 256)

		self.conv1 = nn.Conv2d(1, 32, 4, 2, 1) # shape (batch_size, 32, 128, 128)

		self.conv2 = nn.Conv2d(32, 64, 4, 2, 1, bias=False) # shape (batch_size, 64, 64, 64)
		self.bn2 = nn.BatchNorm2d(64)

		self.conv3 = nn.Conv2d(64, 128, 4, 2, 1, bias=False) # shape (batch_size, 128, 32, 32)
		self.bn3 = nn.BatchNorm2d(128)
		
		self.conv4 = nn.Conv2d(128, 256, 4, 2, 1, bias=False) # shape (batch_size, 256, 16, 16)
		self.bn4 = nn.BatchNorm2d(256)

		self.conv5 = nn.Conv2d(256, 512, 4, 2, 1, bias=False) # shape (batch_size, 512, 8, 8)
		self.bn5 = nn.BatchNorm2d(512)

		self.conv6 = nn.Conv2d(512, 1024, 4, 2, 1, bias=False) # shape (batch_size, 1024, 4, 4)
		self.bn6 = nn.BatchNorm2d(1024)

		self.apply(weights_init)

	def forward(self, x):
		x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
		x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn5(self.conv5(x)), 0.1, inplace=True)
		x = F.leaky_relu(self.bn6(self.conv6(x)), 0.1, inplace=True)

		return x

class DHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv = nn.Conv2d(1024, 1, 4)

		self.apply(weights_init)

	def forward(self, x):
		output = torch.sigmoid(self.conv(x))

		return output.view(-1)

class QHead(nn.Module):
	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(1024, 256, 4, bias=False)
		self.bn1 = nn.BatchNorm2d(256)

		self.conv_disc = nn.Conv2d(256, 100, 1)

		self.conv_mu = nn.Conv2d(256, 2, 1)
		self.conv_var = nn.Conv2d(256, 2, 1)

		self.apply(weights_init)

	def forward(self, x):
		x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

		disc_logits = self.conv_disc(x).squeeze()

		# Not used during training for celeba dataset.
		mu = self.conv_mu(x).squeeze()
		var = torch.exp(self.conv_var(x).squeeze())

		return disc_logits, mu, var
