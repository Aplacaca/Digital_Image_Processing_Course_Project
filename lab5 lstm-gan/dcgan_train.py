# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan.py
@Time    :   2022/05/14 09:42:21
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train a DCGAN model
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from config import DefaultConfig
from dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.setup_seed import setup_seed
from utils.exception_handler import exception_handler
from models.dcgan import Generator, Discriminator

# config
opt = DefaultConfig()

# global config
setup_seed(opt.seed)
torch.cuda.set_device(0)

# mkdir
os.makedirs(opt.result_dir, exist_ok=True)
os.makedirs(opt.save_model_file, exist_ok=True)
os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)


@exception_handler
def train():
    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator(opt)

    if opt.use_gpu:
        generator.to(opt.device)
        discriminator.to(opt.device)
        adversarial_loss.to(opt.device)

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset(img_dir=opt.train_dataset_path + opt.img_class,
                               csv_path='./weather_data/dataset_train.csv',
                               img_size=opt.img_size)
    dataloader = iter(range(len(datasets)))

    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env)

    # ----------
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(datasets), bar_format=bar_format) as bar:
            for i, imgs_index in enumerate(dataloader):
                imgs = datasets[imgs_index]

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:2d}")

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                    1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                    0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(
                    0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(
                    discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                # display the last part of progress bar
                bar.set_postfix_str(
                    f'D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}\33[0m')
                bar.update()

                # visualize the loss curve and generated images in visdom
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='G loss', y=g_loss.item())
                    vis.plot(win='Loss', name='D loss', y=d_loss.item())
                if opt.vis:
                    vis.img(name='Real', img_=imgs.data[:1], nrow=1)
                    vis.img(name='Fake', img_=gen_imgs.data[:1], nrow=1)

                # save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
                    save_image(
                        gen_imgs.data[:9], opt.result_dir+f"{i}.png", nrow=3, normalize=True)
                    torch.save(generator.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + 'generator_'+str(i)+'.pth')
                    torch.save(discriminator.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + 'discriminator_'+str(i)+'.pth')


if __name__ == '__main__':
    train()
