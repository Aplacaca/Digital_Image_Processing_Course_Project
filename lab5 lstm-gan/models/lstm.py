# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan.py
@Time    :   2022/05/14 09:42:21
@Author  :   Dong Huanyu
@Version :   1.0
@Contact :   3057931787@qq.com
@License :   (C)Copyright 2022 Huanyuuu, All rights reserved.
@Desc    :   Train a DCGAN model
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataloader
from models.backbone import FeatureExtractor

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # tensor.contiguous() 会返回有连续内存的相同张量
        # 有时候tensor并不是占用一整块的内存，而是不同的数据块
        # 这个就是tensor在内存中的分布变成连续的，方便后面的view()处理
        # x[:, :, :-self.chomp_size] ，这个就是分块，实现因果卷积
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        output = self.network(state)  # -1, num_channels[-1], tau
        output = output[:, :, -1]  # -1, num_channels[-1]
        # TODO(gpl)
        return output


class GRU(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 batch_first: bool = True,
                 dropout: float = .0):
        super().__init__()
        self.gru = nn.GRU(input_size, output_size, batch_first=batch_first, dropout=dropout)

    def forward(self, state: torch.Tensor):
        state = state.transpose(1, 2)  # (B, input_size, tau) -> (B, tau, input_size)
        return self.gru(state)[1].squeeze()


class MiniGRUCell(nn.Module):
    """Minimal GRU Cell"""

    def __init__(self, input_size: int,
                 hidden_size: int):
        """Initialize layers"""

        super(MiniGRUCell, self).__init__()

        self.Wf = nn.Linear(input_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()

        self.Wh = nn.Linear(input_size, hidden_size)
        self.Uh = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input, hidden):
        """Implement forward propagation for Minimal GRUCell.

        Args:
            input(torch.Tensor): (B,input_size) tensor for input vector
            hidden(torch.Tensor): (B,output_size) tensor for hidden vector

        Returns:
            output(torch.Tensor): (B,output_size) tensor for output vector
        """
        f = self.sigmoid(self.Wf(input) + self.Uf(hidden))
        h = self.tanh(self.Wh(input) + self.Uh(hidden))
        output = (1 - f) * hidden + f * h
        return output


class MGRU(nn.Module):
    """Minimal GRU"""

    def __init__(self, input_size: int,
                 output_size: int):
        super(MGRU, self).__init__()

        self.output_size = output_size
        self.rnn = MiniGRUCell(input_size, output_size)

    def forward(self, state: torch.Tensor):
        state = state.permute(2, 0, 1)  # (B, input_size, tau) -> (tau, B, input_size)
        hx = torch.zeros(state.shape[1], self.output_size, device=state.device)

        # output = []
        for i in range(len(state)):
            hx = self.rnn(state[i], hx)
            # output.append(hx)

        # return torch.stack(output, 0)
        return hx


class LSTM(nn.Module):

    def __init__(self,
                 opt, 
                 input_size: int,
                 output_size: int,
                 batch_first: bool = True,
                 dropout: float = .0):
        super(LSTM, self).__init__()
        self.init_size = opt.img_size // 4
        self.feature = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))
        self.lstm = nn.LSTM(input_size, output_size, batch_first=batch_first, dropout=dropout)

    def forward(self, z):
        out = self.feature(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.lstm(out)
        return img
    
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = int((kernel_size - 1) / 2)

        self.Wxi = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whi = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxf = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whf = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxc = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Whc = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)
        self.Wxo = nn.Conv2d(self.input_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=True)
        self.Who = nn.Conv2d(self.hidden_channels, self.hidden_channels, self.kernel_size, 1, self.padding, bias=False)

        self.Wci = None
        self.Wcf = None
        self.Wco = None

    def forward(self, x, h, c):
        ci = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        cf = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        cc = cf * c + ci * torch.tanh(self.Wxc(x) + self.Whc(h))
        co = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = co * torch.tanh(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        if self.Wci is None:
            self.Wci = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wcf = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
            self.Wco = nn.Parameter(torch.zeros(1, hidden, shape[0], shape[1])).cuda()
        else:
            assert shape[0] == self.Wci.size()[2], 'Input Height Mismatched!'
            assert shape[1] == self.Wci.size()[3], 'Input Width Mismatched!'
        return (Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda(),
                Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])).cuda())



    
if __name__ == '__main__':
    from ..config import TSConfig
    opt = TSConfig()
    inp = torch.randn(20,1,224,224)



