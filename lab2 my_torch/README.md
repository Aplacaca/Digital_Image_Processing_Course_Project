# 任务

本次作业是搭建的简易深度学习框架 `my_torch`，具体内容包括：

- [ ] Modules
    - [x] Linear
    - [ ] Conv2d
- [ ] Functional
    - [x] relu
    - [x] sigmod
    - [x] softmax
    - [ ] MaxPool
    - [ ] AvgPool
    - [ ] Dropout
    - [ ] ……
- [ ] Loss Function
    - [x] MSELoss
    - [x] CrossEntropyLoss
    - [ ] ……
- [ ] Optim
    - [x] SGD
    - [x] Adagrad
    - [x] RMSprop
    - [x] Adam
    - [ ] ……
- [ ] my_tensor 
    - [x] ones
    - [x] zeros
    - [x] zeros_like
    - [x] from_array
    - [x] random
- [ ] Others
    - [ ] cuda
    - [ ] regularization
    - [x] visualize

使用 mytorch 框架，实现 half_moon 数据集二分类问题：`half_moon_mytorch.py` 和 手写数字图片识别`minst_mytorch.py`

用pytorch 框架实现的baseline效果详见：`half_moon_torch.py`、`minst_torch.py`

&nbsp;

# 基本流程

- 开始程序
- 定义网络结构，每层都维护一个参数对象 my_tensor
- 初始化参数设置、生成数据集
- 实例化网络模型 Model、损失函数 Loss Function、优化器 Optim
- 开始训练，循环：
    - 网络Model 前向传播计算预测值
    - Loss Function 对象
        - 计算误差
        - 误差反向传播，链式法则更新每层参数 my_tensor 的梯度属性的值（计算最新梯度值）
    - Optim 对象
        - 按照本轮更新的 my_tensor 的梯度属性，更新每层参数 my_tensor 的 value 属性（更新参数值）
- 循环结束，测试模型效果
- 退出程序

&nbsp;

# 功能模块开发

## 网络模块：Modules

- 定义 mytorch.modules 中的 Linear 层
    - 输入：input_size，out_size
    - 属性：网络权重 w = tensor(input_size+1, output_size)
    - 方法：
        - forward()：计算并保存输入值
        - backward()：根据 `上回输出` 和 `传入的dy` 计算梯度
- 定义 mytorch.functional 中的 Relu、Sigmod 对象
    - forward()：计算并保存输入值
    - backward()：根据 `上回输出` 和 `传入的dy` 计算梯度

## 损失函数：Loss Function

- 定义 mytorch.functional 中的 MSELoss
    - \_\_call_\_()：计算 predict、y 的误差
    - backward()：误差反向传播，网络中每层的 `参数my_tensor` 按照**链式法则**层层更新本轮的 gard 属性值

## 优化器：Optim

- 定义 mytorch.optim 中的 SGD 对象
    - 属性：网络的所有层的 `参数my_tensor`（`参数my_tensor` 包含：value、gard）、lr、momentum
    - 方法：
        - zero_grad()：torch 中计算图的梯度是累积的，需要手动清零，但是我们应该用不到
        - step()：网络所有层按照 lr、`参数my_tensor`本轮的 grad、momentum 等，更新 `参数my_tensor` 的 value

## 仿Tensor对象：my_tensor

- 继承：ndarray，加入 gard 属性，用于 optim 中梯度更新参数
- 属性：

    - value：存放自身值
    - grad：存放梯度值
- 方法：
    - ones
    - zeros
    - zeros_like
    - from_array
    - random

&nbsp;

# Reference

[hello-dian.ai/lab1](https://github.com/npurson/hello-dian.ai/tree/main/lab1)

[pytorch document](https://pytorch.org/docs/stable/index.html)

