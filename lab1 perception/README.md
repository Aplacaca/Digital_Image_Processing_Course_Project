# 线性模型（Linear Model）

## 简介

给定由 `d` 个属性描述的样本 `x=(x1;x2;……;xd)`，其中 `xi` 是 `x` 在第 `i` 个属性的取值

线性模型试图学得属性的线性组合来进行预测的函数，即：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316214732.png" alt="image-20220316214732328" style="zoom:50%;" />

<center>向量形式</center>

参数 w、b 初始值人为给定，由机器学习为最佳值后，模型就得已确定

线性模型形式简单、易于建模，w 直观地表达了各个属性在预测中的重要性，因此具有很强的可解释性（comprehensibility），蕴含机器学习中一些重要思维

许多功能更为强大的非线性模型可以在线性模型的基础上，引入层级结构或高维映射而得

&nbsp;

## 线性回归

### 基本概念

#### 问题

输入：

```
数据集 D={(x1,y1),(x2,y2),……,(xm,ym)}，其中 xi=(x1;x2;……;xd)，yi∈R
```

输出：

```
给定 x=(x1;x2;……;xd)，预测 y*= f(x) = wx+b ≈ y
```

#### 基础知识

1. 无论属性值 `xi` 还是标签 `y`

    - 如果存在“序”关系，则可用一个连续值的大小表示

        例：[高，中，低] = [1，0.5，0]

    - 如果不存在“序”关系，则用**独热向量**表示，不能用连续值的大小表示，否则会导致距离计算产生误差

        例：[红，绿，蓝] = [100，010，001]

        

2. 回归问题中，衡量预测值与实际值差距时，通常使用均方误差：

    <img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316220219.png" alt="image-20220316220219309" style="zoom:50%;" />

    均方误差的几何意义代表着欧式距离

    

3. 最小二乘法

    基于均方误差最小化来进行模型求解的方法就是最小二乘法

    线性回归中，最小二乘法就是试图找到一条直线（即合适的 `w`、`b`），使所有样本到直线上的欧式距离之和最小，这个过程称为 最小二乘参数估计（parameters estimation）

    具体方法就是 Loss 分别对 `w`、`b` 求导，导数为零的点即为所求



### 一元线性回归

一元线性回归即指样本属性值只有**1**个的最简单情况，这种情况下的 Loss 对 `w`、`b` 的导数分别为：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316221230.png" alt="image-20220316221230635" style="zoom:50%;" />

联立求得零点为：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316221450.png" alt="image-20220316221450072" style="zoom:50%;" />



### 多元线性回归

多元线性回归显然是更加一般化的情况，样本有多个属性值

为方便表示，将 `w`、`b`合为一个向量：`w* = (w,b)`，同时 `x` 也做相应变化,最右加入一列全为1的值：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316222122.png" alt="image-20220316222122150" style="zoom:50%;" />

则有：`y = Xw*`

那么优化目标即成为：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316222353.png" alt="image-20220316222353430" style="zoom:50%;" />

这里计算均方误差对 `w*` 一阶导的零点计算涉及矩阵求逆，当 `XTX` 为满秩矩阵（full-rank matrix）或正定矩阵（positive definite matrix）时，零点为为：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316222434.png" alt="image-20220316222433990" style="zoom:50%;" />

然而，现实任务中 `XTX` 往往不是满秩矩阵，甚至可能样本属性的数量还要多于样本的数量，即 x 的列数大于行数，导致有多个 w 解都能使均方误差最小。具体选择哪一个解作为输出将由学习算法的归纳偏好决定，常见做法是引入**正则化**（regularization）



### 对数线性回归 & 广义线性回归

#### 对数回归

对数回归与线性回归的变化在于，不再直接把 `wxi+b` 的值作为 `y` 的预测值，而是在外套了一层指数函数：

![image-20220316223353961](https://gitee.com/lrk612/md_picture/raw/master/img/20220316223354.png)

这样做引入了非线性层，外面这层指数函数（或者说是对数运算）起到了将线性回归模型的预测值**映射到**真实值得作用

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316223515.png" alt="image-20220316223515507" style="zoom:65%;" />



#### 广义线性回归

相当于在对数回归基础上更进一步一般化，不再是用具体的某种函数套在线性回归的结果 `wx+b` 上向真实值 `y` **映射**，而是一般化为**任意**一种**单调可微函数** `g(•)`：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316223744.png" alt="image-20220316223744796" style="zoom:90%;" />

&nbsp;

## 对数几率回归

> 本质是一种广义线性回归，用于解决二分类问题

### 基本思想

为了实现二分类问题，需要对线性回归的结果 `wx+b` 进行量化，映射到 `[0, 1]` 中的某个值

理想的映射是**单位阶跃函数**（unit-step function）：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316224508.png" alt="image-20220316224508850" style="zoom:80%;" />

但是单位阶跃函数在 `x=0` 处不可微，不符合广义线性回归的定义，因此采用**对数几率函数**（logit function）来近似代替：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316224631.png" alt="image-20220316224631081" style="zoom:80%;" />

<center>本质是一种 sigmod 函数</center>

其中 `z=wx+b`，`y` 值就是样本 `x` 作为**正例**的可能性，即：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316230043.png" alt="image-20220316230043137" style="zoom:80%;" />



### 优点

- 直接对分类可能性进行建模，无需事先假设数据分布，避免假设不准确带来的问题
- 不仅是预测出类别，而是得到近似概率预测
- 目标函数是任意阶可导的凸函数，用数值优化算法易求最优解



### 模型求解

> 使用：极大似然法 + 凸优化理论（梯度下降、牛顿法等）

直接上结论（过程容易理解，懒得打了）：最大化：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316230316.png" alt="image-20220316230316756" style="zoom:50%;" />

其中：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316230439.png" alt="image-20220316230439685" style="zoom:50%;" />

<center>p1、p0分别对应预测1、0的概率</center>

代入后进一步使用求导取零点的思路来求解

&nbsp;

## 感知机——二分类











