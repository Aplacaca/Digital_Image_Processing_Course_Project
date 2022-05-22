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

import time
import argparse
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import TSConfig
from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
from models.dcgan import Generator as dc_generator
from models.backbone import FeatureExtractor
from TrainPipeline.dcgan_TrainPipeline import denormalize


parser = argparse.ArgumentParser()
parser.add_argument('--g_path', type=str, default='best/dcgan_radar_generator_90.pth', help='xxx')
parser.add_argument('--fe_path', type=str, default='best/dcgan_radar_fe_90.pth', help='xxx')
parser.add_argument('--lstm_path', type=str, default='best/lstm_preditor_0_7000.pth', help='xxx')
parser.add_argument('--cuda', type=int, default=0, help='xxx')
parser.add_argument('--port', type=int, default=8099, help='xxx')
parse = parser.parse_args()

# config
opt = TSConfig()
if parse.cuda is not None:
    torch.cuda.set_device(parse.cuda)
    opt.use_gpu = True
else:
    opt.use_gpu = False


@exception_handler
def lstm_Test(opt, g_path, fe_path, lstm_path):
    """LSTM Test Pipeline
            
    Parameters:
    ------- 
    opt:
        the config of the model
    g_path:
        the path of the generator
    fe_path:
        the path of the feature extractor
    lstm_path:
        the path of the LSTM
    """
    

    print('LSTM! 🎉🎉🎉')


    # Initialize feature_extractor、generator and discriminator
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)
    predictor = torch.nn.LSTM(input_size=100, hidden_size=100, batch_first=True, num_layers=5)
    
    # Load model
    feature_extractor.load_state_dict(torch.load(fe_path))
    generator.load_state_dict(torch.load(g_path))
    predictor.load_state_dict(torch.load(lstm_path))
    print(f'🌈 模型加载成功！')
    
    # Device
    if opt.use_gpu:
        predictor.to(opt.device)
        feature_extractor.to(opt.device)
        generator.to(opt.device)

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset(img_dir=opt.train_dataset_path + 'Radar', csv_path=opt.train_csv_path, img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=opt.batch_size, shuffle=False, num_workers=0)
    
    print(f'🔋 数据加载成功！')

    # Start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env, port=parse.port)

    # ----------
    #  Testing
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始测试！')

    with tqdm(total=len(dataloader), bar_format=bar_format) as bar:
        for i, imgs in enumerate(dataloader):
            
            # display the first part of progress bar
            bar.set_description(f"\33[36m🌌 ")
            
            # Predict a batch of images features
            history_imgs = Variable(imgs[0:20].type(Tensor))
            history_features = feature_extractor(history_imgs).unsqueeze(0)
            pred_features, _ = predictor(history_features)
            pred_features = pred_features.squeeze(dim=0)
            
            # Generate ground truths features
            future_imgs = Variable(imgs[20:40].type(Tensor), requires_grad=False)
            future_features = feature_extractor(future_imgs).unsqueeze(0)
            future_features = future_features.squeeze(dim=0)
            
            # Visualize
            gen_future_imgs = generator(future_features)
            gen_pred_imgs = generator(pred_features)
            
            imgs_ = denormalize(imgs.data[:1])
            gen_future_img = denormalize(gen_future_imgs.data[:1])
            gen_pred_img = denormalize(gen_pred_imgs.data[:1])
            
            vis.img(name='Real', img_=imgs_, nrow=1)
            vis.img(name='Gen', img_=gen_future_img, nrow=1)
            vis.img(name='Pred', img_=gen_pred_img, nrow=1)

            # display the last part of progress bar
            bar.set_postfix_str('\33[0m')
            bar.update()

            # sleep
            time.sleep(0.1)

if __name__ == '__main__':
    with torch.no_grad():
        lstm_Test(opt, parse.g_path, parse.fe_path, parse.lstm_path)