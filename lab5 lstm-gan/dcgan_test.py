# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan_TrainPipeline.py
@Time    :   2022/05/14 09:42:21
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Test the DCGAN model
"""

import os
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable

from config import DefaultConfig
from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.setup_seed import setup_seed
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor
from models.dcgan import Generator as dc_generator, Discriminator as dc_disciminator


# weight 
fe_weight = 'checkpoints/dcgan/Radar/fe_20000.pth'
g_weight = 'checkpoints/dcgan/Radar/generator_20000.pth'
d_weight = 'checkpoints/dcgan/Radar/discriminator_20000.pth'


# config
opt = DefaultConfig()

# global config
setup_seed(opt.seed)
torch.cuda.set_device(0)

# mkdir
os.makedirs('image_dcgan_test', exist_ok=True)

def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


@exception_handler
def dcgan_Test():

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)
    discriminator = dc_disciminator(opt)

    # load weight
    feature_extractor.load_state_dict(torch.load(fe_weight))
    generator.load_state_dict(torch.load(g_weight))
    discriminator.load_state_dict(torch.load(d_weight))
    print('🌈 模型加载成功！')

    # device
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset(img_dir=opt.train_dataset_path + opt.img_class,
                               csv_path=opt.train_csv_path,
                               img_size=opt.img_size)
    dataloader = iter(range(len(datasets)))

    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env)

    # ----------
    #  Testing
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始测试！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(datasets), bar_format=bar_format) as bar:
            for i, imgs_index in enumerate(dataloader):
                imgs = datasets[imgs_index][:20]

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # # Sample noise as generator input
                # z = Variable(Tensor(np.random.normal(
                #     0, 1, (imgs.shape[0], opt.latent_dim))))
                
                # Extract feature maps from real images
                z = feature_extractor(real_imgs)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                fake_imgs = generator(z)

                # display the last part of progress bar
                bar.set_postfix_str(f'Diff: {sum:.2f}\33[0m')
                bar.update()

                # visualize the generated images in visdom
                if opt.vis:
                    imgs_ = denormalize(imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)
                
                # sleep
                time.sleep(1)


if __name__ == '__main__':
    dcgan_Test()
