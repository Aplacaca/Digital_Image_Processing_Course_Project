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

import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


# class ParseCSV:
#     """解析csv数据集"""

#     def __init__(self, csv_path):
#         self.csv_path = csv_path

#     def __call__(self):
#         """

#         Returns:
#         ------- 
#         data_list: List[List[str]]
#             数据集的路径列表，每组数据包含40张图片的路径
#         """

#         data_list = []
#         csv_data = pd.read_csv(self.csv_path)
#         for line in csv_data.values:
#             data_list.append(line.tolist())

#         return data_list


# class Weather_Dataset(Dataset):

#     def __init__(self, img_dir, csv_path, img_size):
#         self.img_dir = img_dir
#         self.img_size = img_size
#         self.data_list = ParseCSV(csv_path)()  # List[List[str]]
#         self.transform = transforms.Compose([
#             transforms.Resize((self.img_size, self.img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5], std=[0.5])
#         ])

#     def __getitem__(self, index):
#         """

#         Returns:
#         ------- 
#         img: Tensor
#             一段时序图片，序列形状为(40, 1, 224, 224)
#         """

#         img_path_prefix = self.img_dir + '/' + \
#             self.img_dir.split('/')[-1].lower() + '_'
#         img_paths = [img_path_prefix + path for path in self.data_list[index]]

#         # read images
#         try:
#             imgs = [Image.open(img_path) for img_path in img_paths]
#         except:
#             print(img_paths)
#             raise Exception('Error: cannot open image')

#         imgs = list(map(self.transform, imgs))

#         imgs = torch.stack(imgs, dim=0)
#         return imgs

#     def __len__(self):
#         return len(self.data_list)


class ParseCSV:
    """解析csv数据集"""

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __call__(self):
        """

        Returns:
        ------- 
        data_list: List[str]
            数据集的路径列表，把所有图片按行优先顺序放在一个列表中
        """

        data_list = []
        csv_data = pd.read_csv(self.csv_path)
        for line in csv_data.values:
            for column in range(40):
                data_list.append(line[column])

        return data_list


class LSTM_Dataset(Dataset):

    def __init__(self, img_dir, csv_path, img_size, img_num=None):
        """
                
        Parameters:
        ------- 
        img_dir: str
            图片所在的文件夹路径
        csv_path: str
            图片的csv文件路径
        img_size: int
            图片的大小
        img_num: int
            图片的数量，如果不指定，则自动读取csv文件中的数据量
        """
        
        self.img_dir = img_dir
        self.img_size = img_size
        self.data_list = ParseCSV(csv_path)()  # List[str]
        self.img_num = img_num if img_num is not None else len(self.data_list)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        """

        Returns:
        ------- 
        img: Tensor
            按照csv中顺序读取索引为index的图片，形状为(1, img_size, img_size)
        """

        img_path_prefix = self.img_dir + '/' + \
            self.img_dir.split('/')[-1].lower() + '_'
        img_path = img_path_prefix + self.data_list[index]

        # read images
        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            raise Exception('Error: cannot open image')

        # process images
        img = self.transform(img)

        return img

    def __len__(self):
        return self.img_num

class GAN_Dataset(Dataset):

    def __init__(self, img_dir, img_size, img_num=None):
        """
                
        Parameters:
        ------- 
        img_dir: str
            图片所在的文件夹路径
        img_size: int
            图片的大小
        img_num: int
            图片的数量，如果不指定，则自动读取文件夹中所有图片
        """
        
        self.img_dir = img_dir
        self.img_size = img_size
        
        imgs = os.listdir(img_dir+'/')
        self.imgs_path = [os.path.join(img_dir, img) for img in imgs]
        self.img_num = img_num if img_num is not None else len(self.imgs_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __getitem__(self, index):
        """

        Returns:
        ------- 
        img: Tensor
            按照csv中顺序读取索引为index的图片，形状为(1, img_size, img_size)
        """

        img_path = self.imgs_path[index]

        # read images
        try:
            img = Image.open(img_path)
        except:
            print(img_path)
            raise Exception('Error: cannot open image')

        # process images
        img = self.transform(img)

        return img

    def __len__(self):
        return self.img_num


if __name__ == '__main__':
    dataset = GAN_Dataset(img_dir='data/Train/Radar', img_size=256)
    
    dataloader = DataLoader(dataset, batch_size=40, shuffle=True,
                        num_workers=4, drop_last=True)

    for i, data in enumerate(dataloader):
        print(data.shape)
