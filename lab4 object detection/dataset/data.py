import os
import wget
import torch
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms


classes = ['bird', 'car', 'dog', 'lizard', 'turtle']


def read_data(train=False, test=False):
    """读取检测数据集中的图像和标签"""

    data_dir = './dataset/tiny_vid'

    # download dataset
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.listdir(data_dir):
        path = './dataset/tiny_vid/data.zip'
        wget.download('http://xinggangw.info/data/tiny_vid.zip', path)
        if zipfile.is_zipfile(path):
            fz = zipfile.ZipFile(path, 'r')
            for file in fz.namelist():
                fz.extract(file, './dataset')

    images = []
    targets = []

    for i, cls in enumerate(classes):
        dir_name = os.path.join(data_dir, cls)
        f_bbox = open(dir_name+'_gt.txt', 'r')
        if train:
            for num in range(150):
                img = os.path.join(dir_name, f'{num+1:06d}.JPEG')
                img = plt.imread(img)
                img = np.moveaxis(img, -1, 0)
                images.append(torch.tensor(img))

                # target:[x1, y1, x2, y2, class_id]
                target = f_bbox.readline().split(' ')[1:]
                target[-1] = target[-1][:-1]
                target = list(map(float, target))
                # 目标框坐标值归一化
                target = list(map(lambda x: x/img.shape[-1], target))
                # 加入类别id
                target.append(i+1)
                targets.append(target)
        elif test:
            [f_bbox.readline() for _ in range(150)]
            for num in range(150, 180):
                img = os.path.join(dir_name, f'{num+1:06d}.JPEG')
                img = plt.imread(img)
                img = np.moveaxis(img, -1, 0)
                images.append(torch.tensor(img))

                # target:[x1, y1, x2, y2, class_id]
                target = f_bbox.readline().split(' ')[1:]
                target[-1] = target[-1][:-1]
                target = list(map(float, target))
                # 目标框坐标值归一化
                target = list(map(lambda x: x/img.shape[-1], target))
                # 加入类别id
                target.append(i+1)
                targets.append(target)
        else:
            print('data error')
            return

    # [num_img, m ,5], m是数据集的任何图像中边界框的最大数量，这里是1
    targets = torch.tensor(targets).unsqueeze(1)

    return images, targets


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
])


class Data(Dataset):
    def __init__(self, train=False, test=False, transform=transform):
        super(Data, self).__init__()
        self.features, self.labels = read_data(train, test)
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.features[index].float()), self.labels[index]

    def __len__(self):
        return len(self.features)


def load_data(batch_size):
    train_iter = DataLoader(
        Data(train=True), batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(
        Data(test=True), batch_size=batch_size, shuffle=False)

    return train_iter, test_iter
