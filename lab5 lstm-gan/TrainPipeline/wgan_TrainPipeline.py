# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   wgan_TrainPipeline.py
@Time    :   2022/05/14 09:42:21
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train a WGAN-GP model
"""

import os
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image

from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor
from models.wgan_gp import Generator as wgan_generator, Discriminator as wgan_disciminator


# Loss weight for gradient penalty
lambda_gp = 10


def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


# Gradient penalty of WGAN-GP
def compute_gradient_penalty(D, real_samples, fake_samples, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty


@exception_handler
def wgan_TrainPipeline(opt):

    print('WGAN-GP! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    # feature_extractor.load_state_dict(torch.load('checkpoints/dcgan/Radar/fe_20000.pth'))
    generator = wgan_generator(opt, [1, opt.img_size, opt.img_size])
    discriminator = wgan_disciminator([1, opt.img_size, opt.img_size])

    # device
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)

    # Optimizers
    optimizer_fe = torch.optim.SGD(feature_extractor.parameters(), lr=opt.lr_fe)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr_g)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr_d)

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
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(datasets), bar_format=bar_format) as bar:
            # for i, imgs_index in enumerate(dataloader):
            i=0
            while i < 10000:
                # imgs = datasets[imgs_index][:20]
                imgs = datasets[19]

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(
                    0, 1, (imgs.shape[0], opt.latent_dim))))
                
                # Extract feature maps from real images
                z = feature_extractor(real_imgs)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                fake_imgs = generator(z)
                
                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.detach().data, Tensor)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                # Backward and Optimize
                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator and Feature Extractor every n_critic steps
                # -----------------

                optimizer_G.zero_grad()
                optimizer_fe.zero_grad()

                if i % opt.n_critic == 0:

                    # Loss measures generator's ability to fool the discriminator
                    fake_validity = discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()
                    optimizer_fe.step()

                    # display the last part of progress bar
                    bar.set_postfix_str(
                        f'D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}, Diff: {sum:.2f}\33[0m')
                    bar.update()

                    # visualize the loss curve and generated images in visdom
                    if opt.vis and i % 50 == 0:
                        vis.plot(win='Loss', name='G loss', y=g_loss.item())
                        vis.plot(win='Loss', name='D loss', y=d_loss.item())
                    if opt.vis:
                        imgs_ = denormalize(imgs.data[:1])
                        fake_imgs_ = denormalize(fake_imgs.data[:1])
                        vis.img(name='Real', img_=imgs_, nrow=1)
                        vis.img(name='Fake', img_=fake_imgs_, nrow=1)

                    # save the model and generated images every 500 batches
                    if i % opt.sample_interval == 0:
                        fake_imgs_ = denormalize(fake_imgs.data[:9])
                        save_image(fake_imgs_, opt.result_dir + opt.img_class +
                                   '/' + f"{i}.png", nrow=3, normalize=False)
                        torch.save(feature_extractor.state_dict(),
                                   opt.save_model_file + opt.img_class + '/' + f"fe_{i}.pth")
                        torch.save(generator.state_dict(),
                                   opt.save_model_file + opt.img_class + '/' + f'generator_{i}.pth')
                        torch.save(discriminator.state_dict(),
                                   opt.save_model_file + opt.img_class + '/' + f'discriminator_{i}.pth')


if __name__ == '__main__':
    wgan_TrainPipeline()
