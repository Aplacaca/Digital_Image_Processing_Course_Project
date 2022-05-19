# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan_TrainPipeline.py
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
from torch.utils.data import DataLoader

from TrainPipeline.dataset import GAN_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from utils.log import denormalize, save_result_and_model
from models.backbone import FeatureExtractor
from models.dcgan import Generator as dc_generator, Discriminator as dc_disciminator


@exception_handler
def dcgan_TrainPipeline(opt):

    print('DCGAN! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)
    discriminator = dc_disciminator(opt)

    # Load model
    # feature_extractor.load_state_dict(torch.load('checkpoints/dcgan/Radar/fe_7_0.pth'))
    # generator.load_state_dict(torch.load('checkpoints/dcgan/Radar/generator_7_0.pth'))
    # discriminator.load_state_dict(torch.load('checkpoints/dcgan/Radar/discriminator_7_0.pth'))

    # device
    if opt.multi_gpu:
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
        adversarial_loss = torch.nn.DataParallel(adversarial_loss)
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)
        adversarial_loss.to(opt.device)

    # Optimizers
    optimizer_fe = torch.optim.SGD(feature_extractor.parameters(), lr=opt.lr_fe)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = GAN_Dataset(img_dir=opt.train_dataset_path + opt.img_class, img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=40, shuffle=False,
                        num_workers=opt.num_workers, drop_last=True)

    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env)

    # ----------
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    # img = None
    for epoch in range(opt.n_epochs):
        with tqdm(total=len(dataloader), bar_format=bar_format) as bar:
            for i, imgs in enumerate(dataloader):

                # if img is None:
                #     img = torch.cat((datasets[1000].unsqueeze(0), datasets[1000].unsqueeze(0)), dim=0)
                #     imgs = img
                # else:
                #     imgs = img

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                imgs = imgs[:20]
                real_imgs = Variable(imgs.type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                    1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                    0.0), requires_grad=False)
                
                # Extract feature maps from real images
                z = feature_extractor(real_imgs)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                fake_imgs = generator(z)

                # ---------------------------------------
                #  Train Generator and Feature Extractor
                # ---------------------------------------

                if i % 5==0:
                    optimizer_G.zero_grad()
                    optimizer_fe.zero_grad()

                    # Loss measures generator's ability to fool the discriminator
                    fake_validity = discriminator(fake_imgs)
                    g_loss = adversarial_loss(fake_validity, valid)
                    g_loss.backward()
                    
                    optimizer_G.step()
                    optimizer_fe.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                real_loss = adversarial_loss(real_validity, valid)
                fake_loss = adversarial_loss(fake_validity, fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                # display the last part of progress bar
                bar.set_postfix_str(
                    f'D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}, diff: {sum:.2f}\33[0m')
                bar.update()

                # ----------
                # visualize
                # ----------
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='G loss', y=g_loss.item())
                    vis.plot(win='Loss', name='D loss', y=d_loss.item())
                    
                    imgs_ = denormalize(imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)

        # save the model and generated images every 5 epochs
        if epoch % 5 == 0:
            real_imgs = torch.cat(tuple((datasets[i].unsqueeze(0) for i in np.random.choice(datasets.__len__(), size=9, replace=False))), dim=0)
            real_imgs = Variable(real_imgs.type(Tensor))
            save_result_and_model(epoch, 50, opt, real_imgs, feature_extractor, generator, discriminator)

if __name__ == '__main__':
    dcgan_TrainPipeline()
