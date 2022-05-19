#  LSTM-GAN

## Dataset使用说明

### LTSM_Dataset

`dataset.py` 中，LSTM_Dataset类把所有图像组按顺序flat为一维结构，即每40张图片为一组，imgs_paths形状为 `[1, 40*23792]`，_\_getitem__()返回指定index的图片，是已进行归一化处理的。

在赛方提供的数据集中，一行有40张图片，对应一个时序序列。因此DataLoader的batch_size严格设置为40不能改动，不用shuffle**，训练时使用方法如下：

```python
from dataset import LSTM_Dataset
from torch.utils.data import DataLoader

# Configure data loader
datasets = LSTM_Dataset(img_dir='./data/Train/Precip', csv_path='./data/Train.csv', img_size=opt.img_size)

# DataLoader, 注意shuffle和num_workers
dataloader = DataLoader(datasets, batch_size=40, shuffle=True,
                        num_workers=opt.num_workers, drop_last=True)

# Train
for i, imgs in enumerate(dataloader):
    imgs = imgs[:20]
    # 此处 imgs 就是含40张处理后图片中前20张的Tensor，即：[20, 1, image_size, image_size]
    
    # 下面使用imgs进行训练，训练集一共有23792行，所以会迭代23792次
```

### GAN_Dataset

由于`Train.csv`中记录的是时间序列，许多图片重复被使用，一共3W张图片出现80W次，这对于GAN的训练没有必要，所以GAN_Dataset不再从`Train.csv`读取，而是直接从图片文件夹中读取出3W张图片进行300epoch的训练。

&nbsp;

## Backnone使用说明

`models/backbone.py` 中，Feature_Extractor类仿照vgg实现特征提取功能，输入为每个batch中的图片组，输出为长为**latent_dim**的特征向量，供LSTM学习、预测和GAN进行解码生成，使用方法如下：

```python
from models.backbone import FeatureExtractor

# initial
feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)

# device
if opt.use_gpu:
        feature_extractor.to(opt.device)

        # optimizer
optimizer_fe = torch.optim.Adam(feature_extractor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# train
for batch:
    optimizer_fe.zero_grad()
    loss()
    optimizer_fe.step()
```

&nbsp;

## 测试脚本

### gan_test.py

测试GAN的效果，使用visdom逐张测试并显示原图和生成图，使用方法如下：

1. 启动visdom监听8098端口

```bash
python -m visdom.server -p 8098
```

2. 启动测试脚本

```bash
# 测试对Radar训练的dcgan的第50个epoch的效果，用cuad:1、visdom端口号为8098
python gan_test.py --model dcgan --epoch 50 --type Radar --cuda 1 --port 8098
```

