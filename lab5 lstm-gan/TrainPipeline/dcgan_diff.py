# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan_TrainPipeline.py
@Time    :   2022/05/14 09:42:21
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train a DCGAN-DIFF model
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.utils import save_image

from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor
from models.dcgan import Generator as dc_generator, Discriminator as dc_disciminator


def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


@exception_handler
def dcgan_Diff(opt):

    print('DCGAN-DIFF! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    feature_extractor.load_state_dict(torch.load('checkpoints/dcgan/Radar/fe_20000.pth'))
    generator = dc_generator(opt)
    discriminator = dc_disciminator(opt)

    # device
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)
        adversarial_loss.to(opt.device)

    # Optimizers
    optimizer_fe = torch.optim.SGD(feature_extractor.parameters(), lr=opt.lr_fe)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

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
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(datasets), bar_format=bar_format) as bar:
            # for i, imgs_index in enumerate(dataloader):
            i = 0
            while i < 10000:
                
                # Display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # input images
                # real_imgs = datasets[imgs_index][1:21]
                # base_imgs = datasets[imgs_index][0:20]
                real_imgs = datasets[19][1:21]
                base_imgs = datasets[19][0:20]
                real_diff = (real_imgs - base_imgs).clamp(-1, 1)

                # Configure input
                real_imgs = Variable(real_imgs.type(Tensor))
                base_imgs = Variable(base_imgs.type(Tensor))
                real_diff = Variable(real_diff.type(Tensor))
                
                # Extract feature maps from real images
                z = feature_extractor(real_diff)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Adversarial ground truths
                real_label = Variable(Tensor(real_imgs.shape[0], 1).fill_(
                    1.0), requires_grad=False)
                fake_label = Variable(Tensor(real_imgs.shape[0], 1).fill_(
                    0.0), requires_grad=False)

                # -----------------
                #  Train Generator and Feature Extractor
                # -----------------

                optimizer_G.zero_grad()
                # optimizer_fe.zero_grad()
                
                # Generate a batch of images
                fake_diff = generator(z)
                fake_imgs = fake_diff + base_imgs
                fake_imgs = fake_imgs.clamp(-1, 1)
                
                # Loss measures generator's ability to fool the discriminator
                fake_validity = discriminator(fake_imgs)
                g_loss = adversarial_loss(fake_validity, real_label)
                g_loss.backward()
                
                optimizer_G.step()
                # optimizer_fe.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_validity = discriminator(real_imgs)
                fake_validity = discriminator(fake_imgs.detach())
                real_loss = adversarial_loss(real_validity, real_label)
                fake_loss = adversarial_loss(fake_validity, fake_label)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                optimizer_D.step()

                # Display the last part of progress bar
                bar.set_postfix_str(
                    f'D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}, diff: {sum:.2f}\33[0m')
                bar.update()

                # Visualize the loss curve and generated images in visdom
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='G loss', y=g_loss.item())
                    vis.plot(win='Loss', name='D loss', y=d_loss.item())
                if opt.vis:
                    real_imgs_ = denormalize(real_imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    real_diff_ = denormalize(real_diff.data[:1])
                    fake_diff_ = denormalize(fake_diff.data[:1])
                    vis.img(name='Real', img_=real_imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)
                    vis.img(name='Real_Diff', img_=real_diff_, nrow=1)
                    vis.img(name='Fake_Diff', img_=fake_diff_, nrow=1)

                # save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
                    fake_imgs_ = denormalize(fake_imgs.data[:9])
                    save_image(fake_imgs_, opt.result_dir + opt.img_class +
                               '/' + f"{i}.png", nrow=3, normalize=False)
                    # torch.save(feature_extractor.state_dict(),
                    #            opt.save_model_file + opt.img_class + '/' + f"fe_{i}.pth")
                    torch.save(generator.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + f'generator_{i}.pth')
                    torch.save(discriminator.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + f'discriminator_{i}.pth')


if __name__ == '__main__':
    dcgan_Diff()
