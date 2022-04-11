# -.-coding:utf-8 -.-
import tqdm
import torch
from torchnet import meter

import fire
import dataset
import utils
import models
from _config import DefaultConfig

# -------------------------config------------------------
opt = DefaultConfig()
utils.setup_seed(opt.seed)

# --------------------------main--------------------------


def train(**kwargs):
    """
    训练
    """

    # 根据命令行参数更新模型
    opt.parse(kwargs)
    vis = utils.Visualizer(opt.env)

    # step1: model
    model = getattr(models, opt.model)()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        print("device: %s" % opt.device)
        model.to(opt.device)

    # step2: data
    train_data = dataset.DataLoader(opt.batch_size).get_data_loader(
        opt.train_data_root, train=True, num_workers=opt.num_workers)
    val_data = dataset.DataLoader(opt.batch_size).get_data_loader(
        opt.train_data_root, train=False, num_workers=opt.num_workers)

    # step3: loss_fn & optimizer
    criterion = torch.nn.CrossEntropyLoss()
    if opt.use_gpu:
        criterion = criterion.to(opt.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)

    # step4: statistics
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(opt.num_classes)  # 返回一个2×2混淆矩阵
    previous_loss = 1e100

    # step5: train
    model.train()  # 训练模式
    for epoch in range(opt.max_epoch):

        # step5.1: init
        loss_meter.reset()
        confusion_matrix.reset()

        # step5.2: train loop
        for i, (data, label) in enumerate(train_data):
            # step5.2.1: train
            if opt.use_gpu:
                data = data.to(opt.device)
                label = label.to(opt.device)
            optimizer.zero_grad()

            score = model(data)
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            # step5.2.2: statistics
            loss_meter.add(loss.item())
            confusion_matrix.add(score.detach(), label.detach())

            if i % opt.print_freq == opt.print_freq - 1:
                vis.plot('loss', loss_meter.value()[0])
                print('[Epoch %d/%d] [Batch %d/%d] [loss: %.4f]' %
                      (epoch + 1, opt.max_epoch, i + 1, len(train_data), loss.item()))

        # step5.3: save model
        path = opt.model_file + '_%d.pth' % (epoch + 1)
        model.save(path)

        # step5.4: test
        val_cm, val_accuracy = val(model, val_data)
        vis.plot('val_accuracy', val_accuracy)
        vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
            epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()), lr=opt.lr))

        # step5.5: adjust lr
        if loss_meter.value()[0] > previous_loss:
            opt.lr = opt.lr * opt.lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr

        previous_loss = loss_meter.value()[0]


def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息，用以辅助训练
    """

    # 验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(opt.num_classes)
    for i, (data, label) in enumerate(dataloader):
        if opt.use_gpu:
            data = data.to(opt.device)
            label = label.to(opt.device)
        score = model(data)
        confusion_matrix.add(score.detach(), label.detach())

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())

    return confusion_matrix, accuracy


def test(**kwargs):
    """
    测试（inference）
    """

    opt.parse(kwargs)

    # model
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    if opt.use_gpu:
        model = model.to(opt.device)

    # data
    test_data = dataset.DataLoader(opt.batch_size).get_data_loader(
        opt.test_data_root, test=True, num_workers=opt.num_workers)

    # inference
    results = []
    for i, (data, label) in tqdm(enumerate(test_data)):
        if opt.use_gpu:
            data = data.to(opt.device)
        score = model(data)
        # probability = torch.nn.functional.softmax(score, dim=1)
        # batch_result = [(path_,probability_) for path_,probability_ in zip(path, probability)]
        results.extend(score.detach().cpu().tolist())
    utils.write_csv(results, opt.result_file)
    return results


def help():
    utils.help(opt)


if __name__ == '__main__':
    fire.Fire()
