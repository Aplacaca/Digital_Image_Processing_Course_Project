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
    if opt.model == 'LrkNet':
        model = getattr(models, opt.model)()
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
    if opt.model == 'alexnet' or opt.model == 'alexnet_transfer':
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
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
    print('data loaded')

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

    # step4:initi  statistics
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
        cm_value, valid_accuracy = step_test(model, valid_data)
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

        cifar_classes = ('airplane', 'automobile', 'bird', 'cat',
                         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        for i in range(len(cifar_classes)):
            print('accuracy of %10s : %.2f%%' %
                  (cifar_classes[i], 100*(cm_value[i][i]/cm_value[:, i].sum())))

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
    for _, (data, label) in enumerate(dataloader):
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

    # step1: model
    model = getattr(models, opt.model)
    if opt.model == 'LrkNet':
        model = getattr(models, opt.model)()
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
    if opt.model == 'alexnet' or opt.model == 'alexnet_transfer':
        transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
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
    print(cm_value)


def ensemble(**kwargs):

    print('Model ensemble begin')

    def pred(norm, *args):

        predict = torch.zeros_like(args[0])
        for i in range(len(args)):
            if norm:  # normalize
                pred_norm = (args[i].T/torch.max(args[i], dim=1)[0]).T
                predict += pred_norm
            else:
                predict += args[i]

        return predict

    # device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # models
    googlenet = getattr(models, 'googlenet_transfer').to(device).eval()
    googlenet.load_state_dict(torch.load(
        './checkpoints/googlenet_transfer/best_model_86-6.pth'))
    resnet18 = getattr(models, 'resnet18').to(device).eval()
    resnet18.load_state_dict(torch.load(
        './checkpoints/resnet18_transfer/best_model_85-49.pth'))
    resnet34 = getattr(models, 'resnet34').to(device).eval()
    resnet34.load_state_dict(torch.load(
        './checkpoints/resnet34_transfer/best_model_84-99.pth'))
    resnet50 = getattr(models, 'resnet50').to(device).eval()
    resnet50.load_state_dict(torch.load(
        './checkpoints/resnet50_transfer/best_model_87-32.pth'))
    resnet101 = getattr(models, 'resnet101').to(device).eval()
    resnet101.load_state_dict(torch.load(
        './checkpoints/resnet101_transfer/best_model_85-16.pth'))
    resnet152 = getattr(models, 'resnet152').to(device).eval()
    resnet152.load_state_dict(torch.load(
        './checkpoints/resnet152_transfer/best_model_85-94.pth'))
    vgg11 = getattr(models, 'vgg11').to(device).eval()
    vgg11.load_state_dict(torch.load(
        './checkpoints/vgg11_transfer/best_model_87-5.pth'))
    vgg13 = getattr(models, 'vgg13').to(device).eval()
    vgg13.load_state_dict(torch.load(
        './checkpoints/vgg13_transfer/best_model_89-45.pth'))
    vgg16 = getattr(models, 'vgg16').to(device).eval()
    vgg16.load_state_dict(torch.load(
        './checkpoints/vgg16_transfer/best_model_88-98.pth'))
    vgg19 = getattr(models, 'vgg19').to(device).eval()
    vgg19.load_state_dict(torch.load(
        './checkpoints/vgg19_transfer/best_model_87-12.pth'))
    dense121 = getattr(models, 'dense121').to(device).eval()
    dense121.load_state_dict(torch.load(
        './checkpoints/densenet121_transfer/best_model_88-64.pth'))
    dense161 = getattr(models, 'dense161').to(device).eval()
    dense161.load_state_dict(torch.load(
        './checkpoints/densenet161_transfer/best_model_88-51.pth'))
    dense169 = getattr(models, 'dense169').to(device).eval()
    dense169.load_state_dict(torch.load(
        './checkpoints/densenet169_transfer/best_model_88-51.pth'))
    dense201 = getattr(models, 'dense201').to(device).eval()
    dense201.load_state_dict(torch.load(
        './checkpoints/densenet201_transfer/best_model_88-67.pth'))

    # daocat model
    dogcat_model = models.vgg11_dogcat.to(device)
    dogcat_model.load_state_dict(torch.load(
        './DogCat/checkpoints/best_94.pth'))

    print('Model loaded')

    # data enhancement
    opt.batch_size = 64
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    test_dataset = datasets.CIFAR10(
        root=opt.test_data_root, train=False, download=True, transform=transform_aug)
    test_data = DataLoader(dataset=test_dataset,
                           batch_size=opt.batch_size, shuffle=False)
    print('Data loaded')

    # test
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    for i, (data, label) in enumerate(test_data):
        data = data.to(device)
        label = label.to(device)
        dc_data = dc_data.to(device)

        google = googlenet(data)
        r18 = resnet18(data)
        r34 = resnet34(data)
        r50 = resnet50(data)
        r101 = resnet101(data)
        r152 = resnet152(data)
        v11 = vgg11(data)
        v13 = vgg13(data)
        v16 = vgg16(data)
        v19 = vgg19(data)
        d121 = dense121(data)
        d161 = dense161(data)
        d169 = dense169(data)
        d201 = dense201(data)

        predict = pred(False, google, r18, r34, r50, r101, r152, v11,
                       v13, v16, v19, d121, d161, d169, d201)
        confusion_matrix.add(predict.detach(), label.detach())

    cm_value = confusion_matrix.value()
    accuracy = sum([cm_value[i][i] for i in range(opt.num_classes)])
    accuracy = 100. * accuracy / cm_value.sum()

    # display
    print("[total test_accuracy: %.2f%%]" % accuracy)
    cifar_classes = ('airplane', 'automobile', 'bird', 'cat',
                     'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    for i in range(len(cifar_classes)):
        print('accuracy of %10s : %.2f%%' %
              (cifar_classes[i], 100*(cm_value[i][i]/cm_value[:, i].sum())))
    print(cm_value)


def help():
    utils.help(opt)


if __name__ == '__main__':
    fire.Fire()
