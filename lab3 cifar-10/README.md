# 实验任务

## 基础

- [x] 使用 Pytorch 框架，搭建训练文件及工程目录
- [x] 在 MNIST、CIFAR10 数据集上训练网络，观察测试集效果
- [x] 使用不同的网络结构
    - [x] ResNet
    - [x] GoogleNet
    - [x] AlexNet
    - [x] VggNet
    - [x] DenseNet



## 进阶

- [ ] 迁移到 Jittor 框架
- [x] 尽可能提高准确率
- [x] 使用迁移学习
- [x] 自行设计较好效果的网络结构
    - [x] LrkNet



# 训练框架

使用某个深度学习框架时，除了要掌握基本的接口使用，合理组织代码来提高整个工程文件的可读性和可拓展性同样十分重要，因此本次实验我们将合理编排每个目录和代码文件的功能和组成，以达到优雅、美观

## 目录结构

- `Data/`：保存数据集
- `checkpoints/`：保存模型训练后的参数文件
- `models/`：保存网络模型对象
- `utils/`：保存训练和测试时用到的功能函数，例如：可视化
- `main.py`：主文件，训练和测试程序的入口，通过不同的命令来指定操作和参数
- `config.py/`：配置文件，所有可配置的变量都集中在此，并提供默认值
- `README`：提供程序的必要说明
- `requirements.txt`：程序依赖的第三方库

```bash
.
├── Data
│   ├── test
│   │   ├── cifar-10-batches-py
│   │   └── cifar-10-python.tar.gz
│   └── train
│       ├── cifar-10-batches-py
│       └── cifar-10-python.tar.gz
│
├── README.md
├── checkpoints
│	└──…………
│
├── config.py
├── main.py
├── models
│   ├── Basic.py
│   ├── LrkNet.py
│   ├── …………
│   └── __init__.py
│
└── utils
    ├── __init__.py
    ├── utils.py
    ├── help.py
    └── visualize.py
```



## 外部依赖

- **fire**

    2017年3月谷歌开源的一个命令行工具，通过 `pip install fire` 安装

- **meter**

    PyTorchNet 里面的一个工具，提供了一些轻量级的工具，用于帮助用户快速统计训练过程中的一些指标，使用 `pip install torchnet `安装

    `AverageValueMeter`能够计算所有数的平均值和标准差，这里用来统计一个epoch中损失的平均值

    `confusionmeter`用来统计分类问题中的分类情况，返回一个混淆矩阵，可用于计算准确率



## 使用手册

启动可视化服务：`python -m visdom.server`

执行网络训练：`python main.py train --args=xx`

测试网络效果：`python main.py test --model <model_name> -- load_model_path <权重文件路径>`

测试集成学习效果：`python main.py ensemble`

帮助文件：`python main.py help`

