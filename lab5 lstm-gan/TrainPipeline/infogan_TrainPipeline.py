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

from TrainPipeline.dataset import GAN_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor
from models.Infogan import Generator, Discriminator, DHead, QHead


# hyperparameters
noise_dim = 100 # dimension of incompressible noise
discrete_num = 10 # number of discrete latent code used
discrete_dim = 10 # dimension of discrete latent code
continuous_num = 2 # number of continuous latent code used


def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


def noise_sample(discrete_num, discrete_dim, continuous_num, noise_dim, batch_size, device):
    """Sample random noise vector for training.

    Parameters:
    --------
    discrete_num : Number of discrete latent code.
    discrete_dim : Dimension of discrete latent code.
    continuous_num : Number of continuous latent code.
    noise_dim : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, noise_dim, 1, 1, device=device)

    idx = np.zeros((discrete_num, batch_size))
    if(discrete_num != 0):
        dis_c = torch.zeros(batch_size, discrete_num, discrete_dim, device=device)
        
        for i in range(discrete_num):
            idx[i] = np.random.randint(discrete_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if(continuous_num != 0):
        # Random uniform between -1 and 1
        con_c = torch.rand(batch_size, continuous_num, 1, 1, device=device) * 2 - 1

    noise = z
    if(discrete_num != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if(continuous_num != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx


class NormalNLLLoss(torch.nn.Module):
    """
    Calculate the negative log likelihood of normal distribution.
    This needs to be minimised.

    Treating Q(cj | x) as a factored Gaussian.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll


@exception_handler
def info_TrainPipeline(opt):

    print('InfoGAN! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Initialize feature_extractor, generator, discriminator, dhead, qhead
    # feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = Generator()
    discriminator = Discriminator()
    d_head = DHead()
    q_head = QHead()

    # Loss function
    criterion_D = torch.nn.BCELoss()
    criterionQ_dis = torch.nn.CrossEntropyLoss()
    criterionQ_con = NormalNLLLoss()

    # device
    if opt.multi_gpu:
        # feature_extractor = torch.nn.DataParallel(feature_extractor)
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)
        d_head = torch.nn.DataParallel(d_head)
        q_head = torch.nn.DataParallel(q_head)
        criterion_D = torch.nn.DataParallel(criterion_D)
        criterionQ_dis = torch.nn.DataParallel(criterionQ_dis)
        criterionQ_con = torch.nn.DataParallel(criterionQ_con)
    if opt.use_gpu:
        # feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)
        d_head.to(opt.device)
        q_head.to(opt.device)
        criterion_D.to(opt.device)
        criterionQ_dis.to(opt.device)
        criterionQ_con.to(opt.device)

    # Optimizers
    # optimizer_fe = torch.optim.Adam(feature_extractor.parameters(), lr=opt.lr_fe, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam([{'params': discriminator.parameters()}, {'params': d_head.parameters()}], lr=opt.lr_d, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam([{'params': generator.parameters()}, {'params': q_head.parameters()}], lr=opt.lr_g, betas=(opt.b1, opt.b2))

    # Fixed Noise
    noise = torch.randn(100, noise_dim, 1, 1, device=opt.device)
    
    idx = np.arange(discrete_dim).repeat(10)
    discrete_noise = torch.zeros(100, discrete_num, discrete_dim, device=opt.device)
    for i in range(discrete_num):
        discrete_noise[torch.arange(0, 100), i, idx] = 1.0
    discrete_noise = discrete_noise.view(100, -1, 1, 1)

    continuous_noise = torch.rand(100, continuous_num, 1, 1, device=opt.device) * 2 - 1
    fixed_noise = torch.cat([noise, discrete_noise, continuous_noise], dim=1)
    
    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = GAN_Dataset(img_dir=opt.train_dataset_path + opt.img_class, img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=40, shuffle=True,
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

                # -----------
                # Preprocess
                # -----------

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                batch_size = 20
                imgs = imgs[:batch_size]
                real_imgs = Variable(imgs.type(Tensor))

                # # Extract feature maps from real images
                # z = feature_extractor(real_imgs)
                # diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                # sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                noise, idx = noise_sample(discrete_num, discrete_dim, continuous_num, noise_dim, batch_size, opt.device)
                fake_imgs = generator(noise)

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
                # optimizer_fe.zero_grad()

                # Loss measures generator's ability to fool the discriminator
                tmp = discriminator(fake_imgs)
                gen_validity = d_head(tmp)
                gen_loss = criterion_D(gen_validity, valid_label)
                # Loss for discrete latent variable
                q_logits, q_mu, q_var = q_head(tmp)
                target = Variable(torch.LongTensor(idx).to(opt.device))
                dis_loss = 0
                for j in range(discrete_num):
                    dis_loss += criterionQ_dis(q_logits[:, discrete_dim*j : discrete_dim*j+discrete_dim], target[j])
                # Loss for continuous latent variable
                con_loss = 0.1*criterionQ_con(noise[:, noise_dim+discrete_num*discrete_dim:].view(-1, continuous_num), q_mu, q_var)
                # Total loss
                g_loss = gen_loss + dis_loss + con_loss
    
                # Backward and Optimize
                g_loss.backward()
                optimizer_G.step()
                # optimizer_fe.step()

                # --------
                # Logging
                # --------

                with torch.no_grad():
                    fake_imgs = generator(fixed_noise).detach()
                
                # visualize the loss curve and generated images in visdom
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='G loss', y=g_loss.item())
                    vis.plot(win='Loss', name='D loss', y=d_loss.item())
                    imgs_ = denormalize(imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)
                
                # save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
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
                
                # display the last part of progress bar
                bar.set_postfix_str(f'D loss: {d_loss.item():6.3f}, real_loss: {real_loss.item():.2f}, fake_loss: {fake_loss.item():.2f}, G loss: {g_loss.item():6.3f}\33[0m')
                bar.update()
                


if __name__ == '__main__':
    info_TrainPipeline()
