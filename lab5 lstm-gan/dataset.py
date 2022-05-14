# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   dataset.py
@Time    :   2022/05/14 09:42:46
@Author  :   Li Ruikun
@Version :   1.0
@Contact :   1842604700@qq.com
@License :   (C)Copyright 2022 Li Ruikun, All rights reserved.
@Desc    :   Read images from directory
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ParseCSV:
    """解析csv数据集"""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __call__(self):
        """

        Returns:
        ------- 
        data_list: List[List[str]]
            数据集的路径列表，每组数据包含40张图片的路径
        """

        data_list = []
        csv_data = pd.read_csv(self.csv_path)
        for line in csv_data.values:
            data_list.append(line.tolist())

        return data_list


class Weather_Dataset(Dataset):

    def __init__(self, img_dir, csv_path, img_size):
        self.img_dir = img_dir
        self.img_size = img_size
        self.data_list = ParseCSV(csv_path)()  # List[List[str]]  # ! TODO

    def __getitem__(self, index):
        """

        Returns:
        ------- 
        img: Tensor
            一段时序图片，序列形状为(40, 1, 224, 224)
        """

        img_path_prefix = self.img_dir + '/' + \
            self.img_dir.split('/')[-1].lower() + '_'
        img_paths = [img_path_prefix + path for path in self.data_list[index]]

        # read images
        try:
            imgs = [Image.open(img_path) for img_path in img_paths]
        except:
            print(img_paths)
            raise Exception('Error: cannot open image')

        # transform1: resize
        resize = transforms.Resize((self.img_size, self.img_size))
        imgs = list(map(lambda img: resize(img), imgs))
        # transform2: to tensor
        PIL2Tensor = (lambda img: torch.from_numpy(
            np.asarray(img)).unsqueeze(0))
        imgs = list(map(PIL2Tensor, imgs))
        # transform3: zip gray scale
        type = self.img_dir.split('/')[-1].lower()
        type_id = ['precip', 'radar', 'wind'].index(type)
        imgs = list(map(lambda img: img * [10, 70, 35][type_id] / 255, imgs))

        imgs = torch.stack(imgs, dim=0)
        return imgs

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = Weather_Dataset(img_dir='weather_data/train/Precip',
                              csv_path='weather_data/dataset_train.csv',
                              img_size=224)

    for i in range(len(dataset)):
        print(dataset[i][40])
