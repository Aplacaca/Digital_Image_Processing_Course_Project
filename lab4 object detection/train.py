# coding:utf-8
import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import init
from torch import nn, optim, cuda
from torch.backends import cudnn

from ssd import SSD
from loss import SSDLoss
from utils.visdom import Visualizer
from utils.random_seed import setup_seed
from utils.lr_schedule_utils import WarmUpMultiStepLR
from utils.datetime_utils import datetime, time_delta

setup_seed(729)


class TrainPipeline:
    """ 训练 SSD 模型 """

    def __init__(self, dataloader=None, vgg_path: str = None, ssd_path: str = None,
                 lr=5e-4, momentum=0.9, weight_decay=5e-4, batch_size=16, max_epoch=50,
                 use_gpu=True, lr_steps=(200, 300, 400), warm_up_factor=1/3,
                 warm_up_iters=50, **config):
        """
        Parameters
        ----------
        train_loader:
            训练集 DataLoader

        vgg_path: str
            预训练的 VGG16 模型文件路径

        ssd_path: Union[str, None]
            SSD 模型文件路径，有以下两种选择:
            * 如果不为 `None`，将使用模型文件中的参数初始化 `SSD`
            * 如果为 `None`，将使用 `init.xavier` 方法初始化 VGG16 之后的卷积层参数

        lr: float
            学习率

        momentum: float
            动量

        weight_decay: float
            权重衰减

        batch_size: int
            训练集 batch 大小

        max_epoch: int
            训练轮数

        use_gpu: bool
            是否使用 GPU 加速训练

        lr_steps: Tuple[int]
            学习率退火的节点

        warm_up_factor: float
            热启动因子

        warm_up_iters: int
            迭代多少次才结束热启动

        **config:
            先验框生成、先验框和边界框匹配以及 NMS 算法的配置
        """

        self.config.update(config)

        self.dataloader = dataloader
        self.max_epoch = max_epoch
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        if use_gpu and cuda.is_available():
            self.device = torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        self.config = {
            # 通用配置
            'n_classes': 5+1,
            'variance': (0.1, 0.2),

            # 先验框生成配置
            "image_size": 300,
            'steps': [8, 16, 32, 64, 100, 300],
            'feature_maps': [38, 19, 10, 5, 3, 1],
            'min_sizes': [30, 60, 111, 162, 213, 264],
            'max_sizes': [60, 111, 162, 213, 264, 315],
            'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],

            # NMS 配置
            'top_k': 1,
            'nms_thresh': 0.45,
            'conf_thresh': 0.01,

            # 先验框和边界框匹配配置
            'overlap_thresh': 0.5,

            # 困难样本挖掘配置
            'neg_pos_ratio': 3,

            #  设备
            'device': self.device
        }

        # 创建模型
        self.model = SSD(**self.config).to(self.device)

        # 损失函数和优化器
        self.critorion = SSDLoss(**self.config)
        self.optimizer = optim.SGD(
            self.model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)
        self.lr_schedule = WarmUpMultiStepLR(
            self.optimizer, lr_steps, 0.1, warm_up_factor, warm_up_iters)

        # 初始化模型
        if ssd_path:
            self.model.load_state_dict(torch.load(ssd_path))
            print('🧪 成功载入 SSD 模型：' + ssd_path)
        elif vgg_path:
            self.model.vgg.load_state_dict(torch.load(vgg_path))
            self.model.extras.apply(self.xavier)
            self.model.confs.apply(self.xavier)
            self.model.locs.apply(self.xavier)
            print('🧪 成功载入 VGG16 模型：' + vgg_path)
        else:
            raise ValueError("必须指定预训练的 VGG16 模型文件路径")

        self.vis = Visualizer('Train')

    def save(self, epoch):
        """ 保存模型 """

        if not os.path.exists('./checkpoints/'):
            os.mkdir('./checkpoints/')

        self.model.eval()
        path = f'./model_{epoch}.pth'
        torch.save(self.model.state_dict(), path)

        print(f'\n\n🎉 已将当前模型保存到 {path}\n')

    @staticmethod
    def xavier(module):
        """ 使用 xavier 方法初始化模型的参数 """
        if not isinstance(module, nn.Conv2d):
            return

        init.xavier_uniform_(module.weight)
        init.constant_(module.bias, 0)

    def train(self):
        """ 训练模型 """

        print('🚀 开始训练！')

        bar_format = '{desc}{n_fmt:>2s}/{total_fmt:<3s}|{bar}|{postfix}'
        with tqdm(total=math.ceil(len(self.dataloader)), bar_format=bar_format) as bar:

            self.model.train()
            start_time = datetime.now()
            for epoch in range(self.max_epoch):
                loss_his = [[], [], []]
                for i, (images, targets) in enumerate(self.dataloader):
                    self.current_iter = i

                    bar.set_description(
                        f"\33[36m🌌 Epoch{epoch:2d}/{self.max_epoch} Batch")

                    # 预测边界框、置信度和先验框
                    pred = self.model(images.to(self.device))

                    # 计算损失并、反向传播、学习率退火
                    self.optimizer.zero_grad()
                    loc_loss, conf_loss = self.critorion(pred, targets)
                    loss = loc_loss + conf_loss  # type:torch.Tensor
                    loss.backward()
                    self.optimizer.step()
                    # self.lr_schedule.step()

                    # 丰富进度条内容
                    cost_time = time_delta(start_time)
                    bar.set_postfix_str(
                        f'loss: {loss.item():.3f}, loc_loss: {loc_loss.item():.3f}, conf_loss: {conf_loss.item():.3f}, time: {cost_time}\33[0m')
                    bar.update()

                    # 记录
                    loss_his[0].append(loss.item())
                    loss_his[1].append(loc_loss.item())
                    loss_his[2].append(conf_loss.item())

                # 每轮更新进度条
                start_time = datetime.now()
                print(
                    f'    Average loc_loss[{np.mean(loss_his[1]):.3f}]--conf_loss[{np.mean(loss_his[2]):.3f}]--loss[{np.mean(loss_his[0]):.3f}]')
                print('')
                bar.reset()
                self.save(epoch)

                # 可视化训练过程
                self.vis.plot(win='loss',
                              name='loc_loss', y=np.mean(loss_his[1]))
                self.vis.plot(win='loss',
                              name='conf_loss', y=np.mean(loss_his[2]))
                self.vis.plot(win='loss',
                              name='loss', y=np.mean(loss_his[0]))

        torch.cuda.empty_cache()


if __name__ == "__main__":
    # load dataset
    from dataset.data import load_data
    num_classes = 5
    batch_size = 8
    train_loader, _ = load_data(batch_size)

    # train config
    config = {
        'dataloader': train_loader,
        'n_classes': num_classes+1,
        'vgg_path': './vgg16_reducedfc.pth',
        'batch_size': batch_size,
    }

    # train
    train_pipeline = TrainPipeline(**config)
    train_pipeline.train()
