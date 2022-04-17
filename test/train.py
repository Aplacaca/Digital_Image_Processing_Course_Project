from typing import Dict
import jittor as jt
from jittor import nn
import matplotlib.pyplot as plt
import jittor.transform as trans
import numpy as np

from jittor.dataset.cifar import CIFAR10
from jittor.models.resnet import Resnet50

from argparse import ArgumentParser

import nni





def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        outputs = model(inputs)
        loss = nn.cross_entropy_loss(outputs, targets)
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0]))


def test(model, val_loader, epoch):
    model.eval()

    test_loss = 0
    correct = 0
    total_acc = 0
    total_num = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.data, axis=1)
        acc = np.sum(targets.data==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}'.format(epoch, \
                    batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), acc))
    return acc    
    # print ('Total test acc =', total_acc / total_num)


def param_dict():
    parser = ArgumentParser(description="Hyper parameters")
    parser.add_argument('-b','--batch_size', default=64, type=int)
    parser.add_argument('-l','--learning_rate', default=1e-1, type=float)
    parser.add_argument('-e','--epochs', default=10, type=int)
    args = parser.parse_args()
    return vars(args)


def main (prm):
   
    batch_size = prm['batch_size']
    learning_rate = prm['learning_rate']
    momentum = 0.9
    weight_decay = 1e-4
    epochs = prm['epochs']
    
    my_transform = trans.Compose([
        # trans.Resize(224),
        trans.ToTensor()
        ])
    
    train_loader = CIFAR10(train=True, transform=my_transform).set_attrs(batch_size=batch_size, shuffle=True)

    val_loader = CIFAR10(train=False, transform=my_transform)
    val_loader.set_attrs(batch_size=len(val_loader), shuffle=False)

    model = Resnet50()
    optimizer = nn.SGD(model.parameters(), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train(model, train_loader, optimizer, epoch)
        epoch_acc = test(model, val_loader, epoch)
        nni.report_intermediate_result(epoch_acc)
    nni.report_final_result(epoch_acc)
    
        

if __name__=='__main__':
    jt.flags.use_cuda = 1
    # prms = param_dict()
    # import pdb; pdb.set_trace()
    prms = nni.get_next_parameter()
    main(prms)
    