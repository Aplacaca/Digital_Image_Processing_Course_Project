# -.-coding:utf-8 -.-
import torch
import warnings


class DefaultConfig(object):
    model = 'ResNet34'  # 使用的模型，名字必须与models/__init__.py中的名字一致
    optim = 'Adam'  # 优化器
    dataset = 'CIFAR10'  # 数据集名称
    vis = False  # 是否使用visdom可视化
    vis_env = dataset + '-' + model + '-' + optim   # visdom 环境

    train_data_root = './Data/train/'  # 训练集存放路径
    test_data_root = './Data/test/'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载
    num_classes = 10  # 类别数
    seed = 729  # random seed

    batch_size = 200  # batch size
    use_gpu = True  # use GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # available device
    num_workers = 1  # how many workers for loading data
    print_freq = 10  # print info every N batch

    model_file = 'checkpoints/' + model

    max_epoch = 30
    lr = 1e-3  # initial learning rate
    lr_decay = 0.9  # when val_loss increase, lr = lr*lr_decay
    momentum = 0.9
    weight_decay = 1e-3  # 权重衰减(L2正则化)

    def parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        self.model_file = 'checkpoints/' + self.model
        self.vis_env = self.dataset + '-' + self.model + '-' + self.optim

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))
        print()
