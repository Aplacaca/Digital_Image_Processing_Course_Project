# -.- coding:utf-8 -.-
import os
import fire
import torch
from torch.utils.data import DataLoader
from math import sqrt
from torchnet import meter
from torchvision import datasets, transforms

import utils
import models
from config import DefaultConfig

# config
opt = DefaultConfig()
utils.setup_seed(opt.seed)


def train(**kwargs):
    """
    train the model
    """

    # update args
    opt.parse(kwargs)
    if os.path.exists(opt.model_file) == False:
        os.mkdir(opt.model_file)

    # start visualization
    if opt.vis:
        vis = utils.Visualizer(opt.vis_env)

    # step1: model
    model = getattr(models, opt.model)
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        print("device: %s" % opt.device)
        model.to(opt.device)

    # step2: dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(
        root=opt.train_data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(
        root=opt.test_data_root, train=False, download=True, transform=transform)
    cifar_classes = ('airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_data = DataLoader(dataset=train_dataset,
                            batch_size=opt.batch_size, shuffle=True)
    test_data = DataLoader(dataset=test_dataset,
                           batch_size=opt.batch_size, shuffle=True)

    # step3: loss_fn & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if opt.use_gpu:
        criterion = criterion.to(opt.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # step4: statistics
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)  # 返回一个混淆矩阵
    previous_loss = 1e100

    # step5: train
    model.train()  # 训练模式
    for epoch in range(opt.max_epoch):

        # step5.1: init
        loss_meter.reset()
        confusion_matrix.reset()

        # step5.2: train loop
        for i, (data, label) in enumerate(train_data):
            # 打印batch_size张随机的彩色图片
            if opt.vis:
                vis.img(name='batch_pictures', img_=data,
                        nrow=int(sqrt(opt.batch_size)))

            # step5.2.1: train
            if opt.use_gpu:
                data = data.to(opt.device)
                label = label.to(opt.device)

            # step5.2.2: forward
            optimizer.zero_grad()
            predict = model(data)

            # step5.2.3: backward
            loss = criterion(predict, label)
            loss.backward()
            optimizer.step()

            # step5.2.4: statistics
            loss_meter.add(loss.item())
            confusion_matrix.add(predict.detach(), label.detach())

            if i % opt.print_freq == opt.print_freq - 1:
                if opt.vis:
                    vis.plot('loss', loss_meter.value()[0])
                print('[Epoch %d/%d] [Batch %d/%d] [loss: %.4f]' % (epoch + 1,
                      opt.max_epoch, i + 1, len(train_data), loss_meter.value()[0]))

        # step5.3: save model
        path = opt.model_file + '/model_%d.pth' % (epoch + 1)
        torch.save(model.state_dict(), path)
        # model.save(path)

        # step5.4: test
        test_accuracy = test(model, test_data)
        train_accuracy = sum([confusion_matrix.value()[i][i]
                              for i in range(opt.num_classes)])
        train_accuracy = 100. * train_accuracy / confusion_matrix.value().sum()

        if opt.vis:
            vis.plot('test_accuracy', test_accuracy)
        print("[epoch:%d]--[loss:%.4f]--[train_acc: %.2f%%]--[test_acc:%.2f%%]" %
              (epoch, loss_meter.value()[0], train_accuracy, test_accuracy))

        # step5.5: adjust lr
        if loss_meter.value()[0] > previous_loss:
            opt.lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr

        previous_loss = loss_meter.value()[0]


def test(model, dataloader):
    """
    test the model
    """

    model.eval()  # 验证模式

    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    for i, (data, label) in enumerate(dataloader):
        if opt.use_gpu:
            data = data.to(opt.device)
            label = label.to(opt.device)
        predict = model(data)
        confusion_matrix.add(predict.detach(), label.detach())

    cm_value = confusion_matrix.value()

    accuracy = sum([cm_value[i][i] for i in range(opt.num_classes)])
    accuracy = 100. * accuracy / cm_value.sum()

    return accuracy


def help():
    utils.help(opt)


if __name__ == '__main__':
    fire.Fire()
