# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   lstm_TrainPipeline.py
@Time    :   2022/05/19 10:22:07
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Train LSTM model to predict the features of the image
"""

import os
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from TrainPipeline.dataset import LSTM_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.dcgan import Generator as dc_generator
from models.backbone import FeatureExtractor
from TrainPipeline.dcgan_TrainPipeline import denormalize


@exception_handler
def lstm_TrainPipeline(opt, g_path, fe_path):
    """LSTM Train Pipeline
            
    Parameters:
    ------- 
    opt:
        the config of the model
    g_path:
        the path of the generator
    fe_path:
        the path of the feature extractor
    """
    

    print('LSTM! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)
    
    # Load model
    feature_extractor.load_state_dict(torch.load(fe_path))
    generator.load_state_dict(torch.load(g_path))

    # Initialize predictor
    predictor = torch.nn.LSTM(input_size=100, hidden_size=100, batch_first=True, num_layers=5)
    
    # Loss function
    pred_loss = torch.nn.MSELoss()
    
    # Device
    if opt.use_gpu:
        predictor.to(opt.device)
        pred_loss.to(opt.device)
        feature_extractor.to(opt.device)
        generator.to(opt.device)

    # Optimizers
    optimizer_TS = torch.optim.Adam(predictor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets1 = LSTM_Dataset(img_dir=opt.train_dataset_path + 'Precip', csv_path=opt.train_csv_path, img_size=opt.img_size)
    dataloader1 = DataLoader(datasets1, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    datasets2 = LSTM_Dataset(img_dir=opt.train_dataset_path + 'Radar', csv_path=opt.train_csv_path, img_size=opt.img_size)
    dataloader2 = DataLoader(datasets2, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    datasets3 = LSTM_Dataset(img_dir=opt.train_dataset_path + 'Wind', csv_path=opt.train_csv_path, img_size=opt.img_size)
    dataloader3 = DataLoader(datasets3, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    
    # Start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env, port=8099)

    # ----------
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(dataloader1), bar_format=bar_format) as bar:
            for i, (imgs1, imgs2, imgs3) in enumerate(zip(dataloader1, dataloader2, dataloader3)):
                
                # -----------
                # Preprocess
                # -----------

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Predict a batch of images features
                history_imgs = Variable(imgs2[0:20].type(Tensor))
                history_features = feature_extractor(history_imgs).unsqueeze(0)
                pred_features, _ = predictor(history_features)
                pred_features = pred_features.squeeze(dim=0)

                # Generate ground truths features
                future_imgs = Variable(imgs2[20:40].type(Tensor), requires_grad=False)
                future_features = feature_extractor(future_imgs).unsqueeze(0)
                future_features = future_features.squeeze(dim=0)

                # -----------------
                #  Train predictor
                # -----------------

                predictor.zero_grad()

                # Calculate loss, Backward and Optimize
                ts_loss = pred_loss(pred_features, future_features)
                ts_loss.backward()
                optimizer_TS.step()

                # display the last part of progress bar
                bar.set_postfix_str(
                    f'TS loss: {ts_loss.item():.3f}\33[0m')
                bar.update()

                # ----------
                # Visualize
                # ----------
                if opt.vis:
                    vis.plot(win='Loss', name='TS loss', y=ts_loss.item())
                    with torch.no_grad():
                        gen_future_imgs = generator(future_features)
                        gen_pred_imgs = generator(pred_features)
                    imgs_ = denormalize(imgs2.data[:1])
                    gen_future_img = denormalize(gen_future_imgs.data[:1])
                    gen_pred_img = denormalize(gen_pred_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Gen', img_=gen_future_img, nrow=1)
                    vis.img(name='Pred', img_=gen_pred_img, nrow=1)

                # save the model and generated images every 1000 batches
                if i % opt.sample_interval == 0:
                    with torch.no_grad():
                        gen_future_imgs = generator(future_features)
                        gen_pred_imgs = generator(pred_features)
                    gen_pred_img = denormalize(gen_pred_img.data[:9])/255.0
                    gen_future_imgs = denormalize(gen_future_imgs.data[:9])/255.0
                    imgs_ = denormalize(imgs2.data[:9])/255.0
                    save_image(gen_pred_img, opt.result_dir + opt.img_class + '/' + f"pred_{epoch}_{i}.png", nrow=3, normalize=False)
                    save_image(gen_future_imgs, opt.result_dir + opt.img_class + '/' + f"gen_{epoch}_{i}.png", nrow=3, normalize=False)
                    save_image(imgs_, opt.result_dir + opt.img_class + '/' + f"real_{epoch}_{i}.png", nrow=3, normalize=False)
                    torch.save(predictor.state_dict(), opt.save_model_file + opt.img_class + '/' + f'preditor_{epoch}_{i}.pth')
