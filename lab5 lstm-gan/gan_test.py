# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dcgan_TrainPipeline.py
@Time    :   2022/05/14 09:42:21
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Test the selected model
"""

import time
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import DefaultConfig
from TrainPipeline.dataset import GAN_Dataset
from utils.visualize import Visualizer
from utils.log import denormalize
from utils.setup_seed import setup_seed
from utils.exception_handler import exception_handler
from models.backbone import FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='dcgan', help='xxx')
parser.add_argument('--epoch', type=int, default=45, help='xxx')
parser.add_argument('--type', type=str, default='Radar', help='xxx')
parser.add_argument('--cuda', type=int, default=None, help='xxx')
parser.add_argument('--port', type=int, default=8097, help='xxx')
parse = parser.parse_args()

EPOCH = parse.epoch

# weight 
fe_weight = f'checkpoints/dcgan/{parse.type}/fe_{EPOCH}.pth'
g_weight = f'checkpoints/dcgan/{parse.type}/generator_{EPOCH}.pth'
d_weight = f'checkpoints/dcgan/{parse.type}/discriminator_{EPOCH}.pth'


# config
opt = DefaultConfig()
opt.img_class = parse.type
if parse.cuda is not None:
    torch.cuda.set_device(parse.cuda)
    use_gpu = True
else:
    use_gpu = False
opt.parse(dict(
    gan_model=parse.model,
    img_class=parse.type,
    use_gpu=use_gpu,
))

setup_seed(opt.seed)


@exception_handler
def dcgan_Test():

    # Initialize feature_extractor、generator and discriminator
    if opt.gan_model == 'dcgan':
        from models.dcgan import Generator, Discriminator
        feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
        generator = Generator(opt)
        discriminator = Discriminator(opt)
    elif opt.gan_model == 'wgan':
        from models.wgan_gp import Generator, Discriminator
        feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
        generator = Generator(opt, [1, opt.img_size, opt.img_size])
        discriminator = Discriminator(opt, [1, opt.img_size, opt.img_size])

    # load weight
    feature_extractor.load_state_dict(torch.load(fe_weight))
    generator.load_state_dict(torch.load(g_weight))
    discriminator.load_state_dict(torch.load(d_weight))
    print(f'🌈 {opt.gan_model.capitalize()}[{opt.img_class}] 模型加载成功！')

    # device
    if opt.use_gpu:
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)

    # Tensor convertion
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = GAN_Dataset(img_dir=opt.train_dataset_path + opt.img_class, img_size=opt.img_size)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False,
                        num_workers=0, drop_last=True)

    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env+f'-test-{opt.gan_model.capitalize()}[{opt.img_class}]', port=parse.port)

    # ----------
    #  Testing
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始测试 ')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(dataloader), bar_format=bar_format) as bar:
            for i, imgs in enumerate(dataloader):

                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))
                
                # Extract feature maps from real images
                z = feature_extractor(real_imgs)
                diff = [(z[i]-z[i+1]).detach().cpu().data for i in range(z.shape[0]-1)]
                sum = np.array([diff[i].abs().sum() for i in range(len(diff))]).sum()

                # Generate a batch of images
                fake_imgs = generator(z)

                # display the last part of progress bar
                bar.set_postfix_str(f'Diff: {sum:.2f}\33[0m')
                bar.update()

                # visualize the generated images in visdom
                if opt.vis:
                    imgs_ = denormalize(imgs.data[:1])
                    fake_imgs_ = denormalize(fake_imgs.data[:1])
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=fake_imgs_, nrow=1)
                
                # sleep
                time.sleep(0.1)


if __name__ == '__main__':
    with torch.no_grad():
        dcgan_Test()
