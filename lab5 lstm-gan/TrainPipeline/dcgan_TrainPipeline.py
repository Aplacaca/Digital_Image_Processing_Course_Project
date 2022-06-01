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

from TrainPipeline.dataset import Weather_Dataset_2
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from utils.log import denormalize, save_result, save_model
from models.backbone import FeatureExtractor
from models.dcgan import Generator as dc_generator


@exception_handler
def dcgan_TrainPipeline(opt):
    """DCGAN Train Pipeline"""

    print('DCGAN! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Loss function
    mse_loss = torch.nn.L1Loss()

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)

    # Load model
    # feature_extractor.load_state_dict(torch.load('best/wind_fe.pth'))
    # generator.load_state_dict(torch.load('best/wind_generator.pth'))

    # Device
    if opt.multi_gpu:
        feature_extractor = torch.nn.DataParallel(feature_extractor)
        generator = torch.nn.DataParallel(generator)
        mse_loss = torch.nn.DataParallel(mse_loss)
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        mse_loss.to(opt.device)

    # Optimizers
    optimizer_fe = torch.optim.SGD(feature_extractor.parameters(), lr=opt.lr_fe)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset_2(img_dir=opt.train_dataset_path + opt.img_class, csv_path=opt.train_csv_path, img_size=opt.img_size, img_num=opt.batch_size*opt.row_num, shuffle=True)
    dataloader = DataLoader(datasets, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True)

    # Start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env)

    # ----------
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    # Save config
    fp = open(opt.save_model_file + opt.img_class + '/'  + 'config.txt', 'w')
    fp.write(str(opt.__class__.__dict__))
    fp.close()

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(dataloader), bar_format=bar_format) as bar:
            for i, imgs in enumerate(dataloader):

                # -----------
                # Preprocess
                # -----------

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                imgs = imgs
                real_imgs = Variable(imgs.type(Tensor))
                
                # Extract feature maps from real images
                z = feature_extractor(real_imgs)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                fake_imgs = generator(z)

                # ---------------------------------------
                #  Train Generator and Feature Extractor
                # ---------------------------------------

                optimizer_G.zero_grad()
                optimizer_fe.zero_grad()
                
                # Loss = MSE + MS_SSIM
                m_loss = mse_loss(fake_imgs, real_imgs)
                loss = m_loss
                if opt.multi_gpu:
                    loss = loss.mean()
                loss.backward()
                
                optimizer_G.step()
                optimizer_fe.step()

                # Display the last part of progress bar
                bar.set_postfix_str(f'L1 Loss: {m_loss.item():.3f}, diff: {sum:.2f}\33[0m')
                bar.update()

                # ----------
                # Visualize
                # ----------
                if opt.vis:
                    vis.plot(win='Loss', name='L1 Loss', y=m_loss.item())
                    
                    imgs_ = denormalize(imgs.data[:4])
                    fake_imgs_ = denormalize(fake_imgs.data[:4])
                    vis.img(name='Real', img_=imgs_, nrow=2)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=2)

        # save the model and generated images every epochs
        if epoch % 1 == 0:
            real_imgs = torch.cat(tuple((datasets[i].unsqueeze(0) for i in np.random.choice(datasets.__len__(), size=9, replace=False))), dim=0)
            real_imgs = Variable(real_imgs.type(Tensor))
            save_result(epoch, opt, real_imgs, feature_extractor, generator)
        if epoch % 1 == 0:
            save_model(epoch, 10, opt, feature_extractor, generator)

if __name__ == '__main__':
    dcgan_TrainPipeline()
