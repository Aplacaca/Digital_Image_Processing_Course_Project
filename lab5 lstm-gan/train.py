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

import os
import torch
import argparse

from config import DefaultConfig, TSConfig
from utils.setup_seed import setup_seed


# parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dcgan', help='the model to train')
parser.add_argument('--img_class', type=str, default='Radar', help='the image class to generate')
parser.add_argument('--gpu', type=int, default=0, help='gpu id')
parser.add_argument('--seed', type=int, default=729, help='random seed')

# global config
opt = parser.parse_args()
setup_seed(opt.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.cuda.set_device(opt.gpu)


if __name__ == '__main__':

    # gan config
    gan_opt = DefaultConfig()
    gan_opt.parse(dict(
        gan_model=opt.model,
        img_class=opt.img_class,
        vis=False,
        multi_gpu = False
    ))

    # lstm config
    lstm_opt = TSConfig()

    if opt.model == 'dcgan':
        from TrainPipeline.dcgan_TrainPipeline import dcgan_TrainPipeline
        dcgan_TrainPipeline(gan_opt)
    elif opt.model == 'wgan':
        from TrainPipeline.wgan_TrainPipeline import wgan_TrainPipeline
        wgan_TrainPipeline(gan_opt)
    elif opt.model == 'info':
        from TrainPipeline.Infogan_TrainPipeline import info_TrainPipeline
        info_TrainPipeline(gan_opt)
    elif opt.model == 'dcgan_diff':
        from TrainPipeline.dcgan_diff import dcgan_Diff
        dcgan_Diff(gan_opt)
    elif opt.model == 'wgan_diff':
        from TrainPipeline.wgan_diff import wgan_Diff
        wgan_Diff(gan_opt)
    elif opt.model == 'lstm':
        from TrainPipeline.lstm_TrainPipeline import lstm_TrainPipeline
        lstm_TrainPipeline(lstm_opt)
