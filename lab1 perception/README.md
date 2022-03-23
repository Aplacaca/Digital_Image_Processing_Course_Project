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



### 交叉熵损失函数

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220323142821.png" alt="image-20220323142821449" style="zoom:50%;" />

<center>h表述预测值，y为实际值，交叉熵使得h越接近y，损失值越接近0</center>



### 模型求解

直接上结论（过程容易理解，懒得打了）：最大化：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316230316.png" alt="image-20220316230316756" style="zoom:50%;" />

其中：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220316230439.png" alt="image-20220316230439685" style="zoom:50%;" />

<center>p1、p0分别对应预测1、0的概率</center>

但实际上，使用交叉熵损失函数时，难以求解析解，一般使用凸优化理论中的梯度下降、牛顿法等来求解局部最优解

&nbsp;

## 感知机——二分类

### 问题说明

二元线性回归，求解二分类问题：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317190517.png" alt="image-20220317190517444" style="zoom:70%;" />

<center>双属性散点分布如图</center>

### 求解思路

线性拟合：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317190727.png" alt="image-20220317190727454" style="zoom:50%;" />

损失函数：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317191345.png" alt="image-20220317191345586" style="zoom:75%;" />

梯度下降优化：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317191435.png" alt="image-20220317191435734" style="zoom:50%;" />

### 问题&解决

#### 问题一：非线性映射阈值不当

在线性回归的结果基础上，通常套上一个单调递增且可微的非线性函数来映射到目标值，实验PPT中要求：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317191922.png" alt="image-20220317191922102" style="zoom:50%;" />

<center>阈值为 0</center>

而当我这样做了后，训练结果是这样的：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317192137.png" alt="image-20220317192137534" style="zoom:60%;" />

<center>P = 55.56%, acc = 56.00%</center>

显然是欠拟合，但无论改变学习率、训练轮数都没有改善，说明是优化算法本身出问题了

与同学讨论并反复思考后确定，由于将无序的分类目标编码为：[0, 1]，并且在计算损失函数时使用的是欧氏距离，所以非线性映射的阈值应当取为 0 和 1 的均值：0.5

另一种修改方法是，用**独热向量**来编码目标值，然后使用汉明距离、曼哈顿距离或标准化欧式距离来计算损失函数



#### 问题二：batch size 与 学习率

在遇到问题一时，我怀疑是学习时每学习一条样本就更新导致的问题，所以又写了个用全样本批量训练的方法。虽然问题的原因不在于这点，但却在实践中认识到 batch size 与 学习率之间的匹配关系，当 batch size 过大时，如果学习率太高就永远无法收敛到较优解；同样，batch size 较小时，学习率如果也太小就会导致收敛过慢，需要增加训练轮数



### 最终效果

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317195025.gif" alt="show" style="zoom:60%;" />

<center>实时效果</center>

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220317195059.png" alt="image-20220317195059180" style="zoom:60%;" />

<center>loss 值</center>

&nbsp;

## 多分类学习

考虑 N 个类别：`C1, C2,……,Cn`，多分类学习的基本思路是“拆解法”，即将多分类拆分为若干个二分类问题来求解

具体而言，就是先对问题进行拆分，然后为拆出的每个二分类任务训练一个二分类器，在预测时，对这些分类器的预测结果进行集成以获得最终的多分类结果

这里的关键在于如何对多分类任务进行**拆分**，以及如何对多个二分类器进行**集成**（西瓜书里主要介绍了怎么拆分）



### OvO

一对一拆分，即N个类别两两组成一个二分类器，共 `N(N-1)/2` 个

每次预测都用所有二分类器预测得到的  `N(N-1)/2`  个结果进行计数投票，被预测得最多的类别即为最终分类结果



### OvR

一对多拆分，即每次将一个类别作为正例，其他全作为反例，以此来训练共 `N` 个分类器

预测时若仅有一个分类器预测为正类，则对应的类别标记作为最终分类结果；若有多个分类器预测为正类，则需要考虑各分类器的置信度，取最大者的类别标记作为分类结果

显然，OvR 训练的二分类器数量要少于 OvO，但是 OvR 训练使用的样本数量也更多，所以二者时间上各有优劣，实际效果也差不太多

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220319220824.png" alt="image-20220319220816951" style="zoom:67%;" />

<center>OvO 和 OvR 的预测过程</center>



### MvM

MvM 是每次将若干个类作为正类，其余类作为反类，由此训练多个二分类器。显然，OvO、OvR 都是 MvM 的特殊情况

MvM 的正、反类构造必须有特殊的设计，不能随便选取。一种最常用的 MvM 技术是**纠错输出码**（Error Correcting Output Codes，简称 **ECOC**）

**ECOC**

ECOC 把编码的思想引入类别拆分，并尽可能在解码过程中具有容错性，主要分为两步：

- 编码

    对 N 个类别做 M 次划分，每次划分将一部分类别划分为正类，另一部分划分为反类。从而形成 M 个二分类训练集，对应训练出 M 个二分类器

- 解码

    M 个分类器分别对测试样本进行预测，这些预测标记组成一个编码，计算此编码与每个类别的标准编码的汉明距离或欧氏距离，取距离最小者作为最终预测结果



类别划分通过**编码矩阵**（coding matrix）指定，编码矩阵有多种形式，常见的主要有**二元码**和**三元码**，后者除了正、反类之外，还引入了“停用类”，即不考虑的类项，示例如下：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220319225534.jpg" alt="20220319_225419" style="zoom:15%;" />

<center>可见，即使某个分类器出错带来误差，依然能够输出正确结果</center>

对同等长度的编码，理论上任意两个类别之间的编码距离（汉明或欧式）越远，纠错能力就越强，这就是最优编码（是数学上可求的，但是是NP-难问题）。ECOC 编码越长，纠错能力也越强，但是就越难计算出最优编码

实际上，ECOC 编码较长时，没必要一定使用最优编码，因为最优编码虽然纠错能力最强，但是最优编码所形成的二分类器集未必比次优编码的二分类器集更容易训练

&nbsp;

## 类别不平衡问题

但训练集中正负样本数量均衡时，分类器可以正常训练，但是如果正负样本数量相差较大时，模型的训练可能难以收敛或者达到较好的效果

通常用训练集中样本分布推断实际正反例分布，当训练集中正样本数量为 m+，负样本数量为 m-时，预测结果满足：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220319231923.png" alt="image-20220319231923579" style="zoom:60%;" />

<center>y 为预测为正例的概率</center>

则判定为为正例，反之为反例

但是因为分类器的决策规则实际为：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220319232210.png" alt="image-20220319232210528" style="zoom:60%;" />

时判为正例，所以引入类别不平衡学习的一个基本策略——**再缩放**（rescaling）：

<img src="https://gitee.com/lrk612/md_picture/raw/master/img/20220319232348.png" alt="image-20220319232348534" style="zoom:60%;" />

---

再缩放的思想简单，但是是建立在**训练集是真实样本总体的无偏采样**这一假设上的，实际上找个假设往往并不成立，也就是说未必能够有效地基于训练集观测几率来推断出真实几率，通常采用下面两种方法来处理：

### 过采样

对数量较少的类别样本进行过采样，即增加该类别样本的数量。但是不能简单地复制，否则会导致对该类别样本过拟合。通常用插值等方法来额外扩充

### 欠采样

对数量较多的类别样本进行欠采样，即减少该类别样本的数量。同样不能直接删除某些样本，这样可能会损失重要信息，通常使用一些集成学习机制，将该类别划分为若干个集合供不同学习器使用。这样对每个学习器而言都做了欠采样，但全局上并没有丢失重要信息









