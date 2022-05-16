# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   train.py
@Time    :   2022/05/16 18:50:12
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train the selected model
"""

import torch
import argparse

from config import DefaultConfig
from utils.setup_seed import setup_seed

from TrainPipeline.dcgan_TrainPipeline import dcgan_TrainPipeline
from TrainPipeline.wgan_TrainPipeline import wgan_TrainPipeline
from TrainPipeline.dcgan_diff import dcgan_Diff
from TrainPipeline.wgan_diff import wgan_Diff

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dcgan', help='the model to train')
parser.add_argument('--img_class', type=str, default='Radar', help='the image class to generate')

# global config
setup_seed(729)
torch.cuda.set_device(0)

if __name__ == '__main__':
    opt = parser.parse_args()

    # config
    train_opt = DefaultConfig()
    train_opt.parse(dict(gan_model=opt.model, img_class=opt.img_class, vis=True))

    if opt.model == 'dcgan':
        dcgan_TrainPipeline(train_opt)
    elif opt.model == 'wgan':
        wgan_TrainPipeline(train_opt)
    elif opt.model == 'dcgan_diff':
        dcgan_Diff(train_opt)
    elif opt.model == 'wgan_diff':
        wgan_Diff(train_opt)