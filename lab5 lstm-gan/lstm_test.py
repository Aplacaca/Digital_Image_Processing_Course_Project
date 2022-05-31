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
from TrainPipeline.dataset import Weather_Dataset, TEST_Dataset
from utils.visualize import Visualizer
from utils.log import save_result, save_model
from utils.exception_handler import exception_handler
from models.dcgan import Conv_Generator as dc_generator
from models.backbone import FeatureExtractor
from TrainPipeline.dcgan_TrainPipeline import denormalize


parser = argparse.ArgumentParser()
parser.add_argument('--g_path', type=str, default='best/convgan/generator_10.pth', help='xxx')
# parser.add_argument('--fe_path', type=str, default='best/dcgan_radar_fe_90.pth', help='xxx')
# parser.add_argument('--lstm_path', type=str, default='best/lstm_preditor_0_7000.pth', help='xxx')
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
def lstm_Test(opt, g_path):
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


    # Initialize 
    generator = dc_generator(opt)
    
    # Load model
    generator.load_state_dict(torch.load(g_path))
    print(f'🌈 模型加载成功！')
    
    # Device
    if opt.use_gpu:
        generator.to(opt.device)

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = TEST_Dataset(img_dir='./data/TestB1/Radar', img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=0)
    
    print(f'🔋 数据加载成功！')

    # Start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env, port=parse.port)

    # ----------
    #  Testing
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始测试！')

    with tqdm(total=104, bar_format=bar_format) as bar:
        for i, imgs in enumerate(dataloader):
            
            # display the first part of progress bar
            bar.set_description(f"\33[36m🌌 ")
            
            # Predict a batch of images features
            history_imgs = Variable(imgs.type(Tensor)).squeeze(0)
            # import ipdb;ipdb.set_trace()
            # real_imgs = Variable(imgs[20:].type(Tensor))
            
            gen_pred_imgs = generator(history_imgs)
            
            # real_imgs = denormalize(real_imgs.data)
            history_imgs = denormalize(history_imgs.data)
            gen_pred_img = denormalize(gen_pred_imgs.data)
            # vis.img(name='Real', img_=real_imgs[:4], nrow=2)
            vis.img(name='His', img_=history_imgs[:4], nrow=2)
            vis.img(name='Fake', img_=gen_pred_img[:4], nrow=2)
            if i%5 == 0:
                save_result(i, opt, history_imgs, gen_pred_img, None, generator)
            # display the last part of progress bar
            bar.set_postfix_str('\33[0m')
            bar.update()

            # sleep
            time.sleep(0.1)

if __name__ == '__main__':
    with torch.no_grad():
        lstm_Test(opt, parse.g_path)