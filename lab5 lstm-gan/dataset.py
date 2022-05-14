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
            数据集的路径列表，每组数据包含41张图片的路径
        """

        data_list = []
        csv_data = pd.read_csv(self.csv_path)
        for line in csv_data.values:
            data_list.append(line.tolist())

        return data_list


class Weather_Dataset(Dataset):

    def __init__(self, img_dir, csv_path, img_size):
        self.img_dir = img_dir
        self.data_list = ParseCSV(csv_path)()  # List[List[str]]  # ! TODO
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def __getitem__(self, index):
        """

        Returns:
        ------- 
        img: Tensor
            一段时序图片，序列形状为(41, 1, 224, 224)
        """

        img_path_prefix = self.img_dir + '/' + \
            self.img_dir.split('/')[-1].lower() + '_'

        img_paths = [img_path_prefix + path for path in self.data_list[index]]

        try:
            imgs = [Image.open(img_path) for img_path in img_paths]
        except:
            print(img_paths)
            raise Exception('Error: cannot open image')

        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

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
