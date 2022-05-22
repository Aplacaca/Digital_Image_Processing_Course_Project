# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   Infogan_TrainPipeline.py
@Time    :   2022/05/18 08:23:24
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train a InfoGAN model
"""

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor
from models.Infogan import Generator, Discriminator, DHead, QHead


# hyperparameters
discrete_dim = 100 # dimension of discrete latent code (feature)
noise_dim = 100 # dimension of incompressible noise


def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


@exception_handler
def info_TrainPipeline(opt):

    print('InfoGAN! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Initialize feature_extractor, generator, discriminator, dhead, qhead
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = Generator()
    discriminator = Discriminator()
    d_head = DHead()
    q_head = QHead()

    # Loss function
    criterion_D = torch.nn.BCELoss()
    criterionQ_dis = torch.nn.CrossEntropyLoss()

    # device
    if opt.multi_gpu:
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
        d_head = torch.nn.DataParallel(d_head)
        q_head = torch.nn.DataParallel(q_head)
        criterion_D = torch.nn.DataParallel(criterion_D)
        criterionQ_dis = torch.nn.DataParallel(criterionQ_dis)
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)
        d_head.to(opt.device)
        q_head.to(opt.device)
        criterion_D.to(opt.device)
        criterionQ_dis.to(opt.device)

    # Optimizers
    optimizer_fe = torch.optim.Adam(feature_extractor.parameters(), lr=opt.lr_fe, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam([{'params': discriminator.parameters()}, {'params': d_head.parameters()}], lr=opt.lr_d, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam([{'params': generator.parameters()}, {'params': q_head.parameters()}], lr=opt.lr_g, betas=(opt.b1, opt.b2))
    
    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset(img_dir=opt.train_dataset_path + opt.img_class, csv_path=opt.train_csv_path, img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=opt.batch_size, shuffle=True,
                        num_workers=opt.num_workers, drop_last=True)

    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env)

    # Incompressible noise
    noise = torch.randn(20, noise_dim, 1, 1, device=opt.device) 

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

                # -----------
                # Preprocess
                # -----------

                # Display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                imgs = imgs[:20]
                real_imgs = Variable(imgs.type(Tensor))

                # Extract feature maps from real images
                feature = feature_extractor(real_imgs)
                
                # Input = noise + feature
                feature = feature.view(feature.size(0), feature.size(1), 1, 1)
                input = torch.cat([feature, noise], dim=1)

                # Generate a batch of images
                fake_imgs = generator(input)

                # Adversarial ground truths
                valid_label = Variable(Tensor(imgs.shape[0],).fill_(1.0), requires_grad=False)
                fake_label = Variable(Tensor(imgs.shape[0],).fill_(0.0), requires_grad=False)

                # -------------------------------
                #  Train Discriminator and DHead
                # -------------------------------

                optimizer_D.zero_grad()
                
                # Real images
                tmp = discriminator(real_imgs)
                real_validity = d_head(tmp)
                real_loss = criterion_D(real_validity, valid_label)
                
                # Fake images
                tmp = discriminator(fake_imgs.detach())
                fake_validity = d_head(tmp)
                fake_loss = criterion_D(fake_validity, fake_label)
                
                # Total loss
                d_loss = (real_loss + fake_loss) / 2
                
                # Backward and Optimize
                d_loss.backward()
                optimizer_D.step()

                # ---------------------------
                #  Train Generator and QHead
                # ---------------------------

                optimizer_G.zero_grad()
                optimizer_fe.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                tmp = discriminator(fake_imgs)
                gen_validity = d_head(tmp)
                gen_loss = criterion_D(gen_validity, valid_label)
                
                # Loss for discrete latent variable
                q_logits, q_mu, q_var = q_head(tmp)
                mutual_info_loss = criterionQ_dis(q_logits, feature.squeeze())

                # Total loss
                g_loss = gen_loss + mutual_info_loss
    
                # Backward and Optimize
                g_loss.backward()
                optimizer_G.step()
                optimizer_fe.step()

                # --------
                # Logging
                # --------
                
                # Visualize the loss curve and generated images in visdom
                if opt.vis and i % 5 == 0:
                    with torch.no_grad():
                        fake_imgs = generator(input).detach()
                    vis.plot(win='Loss', name='G loss', y=g_loss.item())
                    vis.plot(win='Loss', name='D loss', y=d_loss.item())
                    imgs_ = denormalize(imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)
                
                # Save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
                    with torch.no_grad():
                        fake_imgs = generator(input).detach()
                    real_imgs_ = denormalize(real_imgs.data[:9])/255.0
                    fake_imgs_ = denormalize(fake_imgs.data[:9])/255.0
                    save_image(real_imgs_, opt.result_dir + opt.img_class +
                               '/' + f"{epoch}_{i}_real.png", nrow=3, normalize=False)
                    save_image(fake_imgs_, opt.result_dir + opt.img_class +
                               '/' + f"{epoch}_{i}_fake.png", nrow=3, normalize=False)
                    torch.save({
                        'generator' : generator.state_dict(),
                        'discriminator' : discriminator.state_dict(),
                        'q_head' : q_head.state_dict(),
                        'd_head' : d_head.state_dict(),
                        # 'feature_extractor' : feature_extractor.state_dict(),
                        'config' : opt
                        }, opt.save_model_file + opt.img_class + '/' + f"{epoch}_{i}.pth")

                # Display the last part of progress bar
                bar.set_postfix_str(f'D loss: {d_loss.item():6.3f}, real_loss: {real_loss.item():.2f}, fake_loss: {fake_loss.item():.2f}, G loss: {g_loss.item():6.3f}\33[0m')
                bar.update()
                


if __name__ == '__main__':
    info_TrainPipeline()
