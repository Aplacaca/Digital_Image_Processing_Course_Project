#  LSTM-GAN

## Dataset使用说明

`dataset.py` 中，Weather_Dataset类读取指定位置的图像，但与一般Dataset不同的是，_\_getitem__()返回的不是一张图片，而是**一行图片**。在赛方提供的数据集中，一行有40张图片，对应一个时序序列。因此DataLoader**不需要shuffle**，训练时使用方法如下：

```python
from dataset import Weather_Dataset

# Configure data loader
datasets = Weather_Dataset(img_dir='./weather_data/Train/Precip', csv_path='./weather_data/dataset_train.csv', img_size=opt.img_size)
dataloader = iter(range(len(datasets)))

# Train
for i, imgs_index in enumerate(dataloader):
    imgs = datasets[imgs_index]
    # 此处 imgs 就是含40张处理后图片的Tensor，即：[40, 1, image_size, image_size]
    
    # 下面使用imgs进行训练，训练集一共有23792行，所以会迭代23792次
```

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

