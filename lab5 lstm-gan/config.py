# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   config.py
@Time    :   2022/05/14 09:43:37
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Configurations for training
"""

import torch
import warnings


class DefaultConfig(object):

    # GAN模型
    gan_model = 'dcgan'

    # 文件路径
    img_class = 'Radar'
    train_csv_path = '/home/lrk/lab5/data/Train.csv'  # csv file path
    train_dataset_path = '/home/lrk/lab5/data/Train/'  # image file path
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    save_model_file = 'checkpoints/'  # weights file path
    result_dir = 'images_' + gan_model + '/'  # result image path

    # 图片参数
    channels = 1  # number of image channels
    img_size = 256  # size of each image dimension

    # 训练参数
    n_epochs = 1
    lr = 2e-4  # initial learning rate
    latent_dim = 100  # dimensionality of the latent space
    b1 = 0.5  # adam: decay of first order momentum of gradient
    b2 = 0.999  # adam: decay of first order momentum of gradient

    # 其他参数
    vis = True  # use visdom
    vis_env = 'LSTM-GAN'   # visdom env
    seed = 729  # random seed
    use_gpu = True  # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # available device
    num_workers = 1  # how many workers for loading data
    sample_interval = 1000  # print info every N batch

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        print()

class TSConfig(object):

    # GAN模型
    ts_model = 'LSTM'

    # 文件路径
    img_class = 'Radar'
    train_csv_path = '/home/lrk/lab5/data/Train.csv'  # csv file path
    train_dataset_path = '/home/lrk/lab5/data/Train/'  # image file path
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    save_model_file = 'checkpoints/'  # weights file path
    result_dir = 'images_' + ts_model + '/'  # result image path

    # 图片参数
    channels = 1  # number of image channels
    img_size = 256  # size of each image dimension

    # 训练参数
    n_epochs = 1
    lr = 2e-4  # initial learning rate
    latent_dim = 100  # dimensionality of the latent space
    b1 = 0.5  # adam: decay of first order momentum of gradient
    b2 = 0.999  # adam: decay of first order momentum of gradient

    # 其他参数
    vis = True  # use visdom
    vis_env = 'LSTM'   # visdom env
    seed = 729  # random seed
    use_gpu = True  # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # available device
    num_workers = 1  # how many workers for loading data
    sample_interval = 1000  # print info every N batch

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        print()
