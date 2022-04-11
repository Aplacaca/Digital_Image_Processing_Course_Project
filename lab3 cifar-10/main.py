# -.- coding:utf-8 -.-
import os
import fire
import torch
from torch.utils.data import DataLoader
from torch.utils.data import sampler
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
    if os.path.exists('checkpoints/') == False:
        os.mkdir('checkpoints/')
    if os.path.exists(opt.model_file) == False:
        os.mkdir(opt.model_file)

    # start visualization
    if opt.vis:
        vis = utils.Visualizer(opt.vis_env)

    # step1: model
    model = getattr(models, opt.model)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path))
    if opt.use_gpu:
        print("device: %s" % opt.device)
        model.to(opt.device)

    # step2: data enhancement
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = datasets.CIFAR10(
        root=opt.train_data_root, train=True, download=True, transform=transform_aug)
    test_dataset = datasets.CIFAR10(
        root=opt.test_data_root, train=False, download=True, transform=transform_aug)

    train_num = int(0.8 * len(train_dataset))  # 训练集:验证集=8:2
    train_data = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                            sampler=sampler.SubsetRandomSampler(range(train_num)))
    valid_data = DataLoader(dataset=train_dataset, batch_size=opt.batch_size,
                            sampler=sampler.SubsetRandomSampler(range(train_num, len(train_dataset))))
    test_data = DataLoader(dataset=test_dataset,
                           batch_size=opt.batch_size, shuffle=False)

    # step3: loss_fn & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if opt.use_gpu:
        criterion = criterion.to(opt.device)
    if opt.optim == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
    elif opt.optim == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    elif opt.optim == 'Adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        raise Exception("optim error")

    # step4: statistics
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)  # 返回一个混淆矩阵
    previous_loss = 1e100

    # step5: train
    model.train()  # 训练模式
    best_acc = 0
    best_model = None
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

        # step5.3: validate
        _, valid_accuracy = step_test(model, valid_data)
        train_accuracy = sum([confusion_matrix.value()[i][i]
                              for i in range(opt.num_classes)])
        train_accuracy = 100. * train_accuracy / confusion_matrix.value().sum()

        if best_acc < valid_accuracy:
            best_acc = valid_accuracy
            best_model = model.state_dict()

        if opt.vis:
            vis.plot('valid_accuracy', valid_accuracy)
        print("[epoch:%d]--[loss:%.4f]--[train_acc: %.2f%%]--[valid_acc:%.2f%%]" %
              (epoch + 1, loss_meter.value()[0], train_accuracy, valid_accuracy))

        # step5.5: lr decay
        if loss_meter.value()[0]/previous_loss > 0.95 and opt.lr > 1e-5:
            opt.lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr
            print('learning rate decay to %f' % opt.lr)

        previous_loss = loss_meter.value()[0]

        # step5.6: save model
        if (epoch+1) % 30 == 0:
            path = opt.model_file + '/model_%d.pth' % (epoch + 1)
            torch.save(best_model, path)

    # step6: save model
    path = opt.model_file + '/best_model_%s-%s.pth' % \
        (str(best_acc).split('.')[0], str(best_acc).split('.')[1])
    print('best_acc: %.2f%%' % best_acc, 'save model to %s' % path)
    torch.save(best_model, path)

    # step7: test
    model.load_state_dict(torch.load(path))
    _, test_accuracy = step_test(model, test_data)
    print('best model valid_acc: %.2f%%' %
          best_acc, 'test_acc: %.2f%%' % test_accuracy)


def step_test(model, dataloader):
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

    model.train()  # 回到训练模式
    return cm_value, accuracy


def test(**kwargs):
    # update args
    opt.parse(kwargs)
    if os.path.exists('checkpoints/') == False:
        os.mkdir('checkpoints/')
    if os.path.exists(opt.model_file) == False:
        os.mkdir(opt.model_file)

    # start visualization
    if opt.vis:
        vis = utils.Visualizer(opt.vis_env)

    # step1: model
    model = getattr(models, opt.model)
    if opt.load_model_path:
        model.load_state_dict(torch.load(opt.load_model_path))
    if opt.use_gpu:
        print("device: %s" % opt.device)
        model.to(opt.device)

    # step2: dataset
    # 数据增强
    transform_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),  # 依概率将图像水平翻转 概率默认为0.5
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_dataset = datasets.CIFAR10(
        root=opt.test_data_root, train=False, download=True, transform=transform_aug)
    test_data = DataLoader(dataset=test_dataset,
                           batch_size=opt.batch_size, shuffle=False)

    # step3: test
    cm_value, test_accuracy = step_test(model, test_data)
    print("[total test_accuracy: %.2f%%]" % test_accuracy)
    cifar_classes = ('airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(len(cifar_classes)):
        print('accuracy of %10s : %.2f%%' %
              (cifar_classes[i], 100*(cm_value[i][i]/cm_value[:, i].sum())))


def help():
    utils.help(opt)


if __name__ == '__main__':
    fire.Fire()
