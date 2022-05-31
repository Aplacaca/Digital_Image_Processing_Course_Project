# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   log.py
@Time    :   2022/05/18 15:28:38
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Record the training process
"""

import torch
from torchvision.utils import save_image


def denormalize(imgs, mean=0.5, variance=0.5):
    return imgs.mul(variance).add(mean) * 255.0


def save_result(epoch, opt, imgs, input_imgs,feature_extractor, generator):

    # save result
    with torch.no_grad():
        real_imgs = imgs
        # z = feature_extractor(real_imgs)
        # fake_imgs = generator(input_imgs)
        fake_imgs = input_imgs
    real_imgs_ = real_imgs.data[:9]/255.0
    fake_imgs_ = fake_imgs.data[:9]/255.0
    save_image(real_imgs_, opt.result_dir + opt.img_class + '/' + f"real_{epoch}.png", nrow=3, normalize=False)
    save_image(fake_imgs_, opt.result_dir + opt.img_class + '/' + f"fake_{epoch}.png", nrow=3, normalize=False)
    
    
def save_model(epoch, opt, feature_extractor, generator, discriminator):

    # save model
        
    # torch.save(feature_extractor.state_dict(), opt.save_model_file + opt.img_class + '/' + f"fe_{epoch}.pth")
    torch.save(generator.state_dict(), opt.save_model_file + opt.img_class + '/' + f'generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), opt.save_model_file + opt.img_class + '/' + f'discriminator_{epoch}.pth')