import jittor as jt
from jittor import nn
import matplotlib.pyplot as plt
import jittor.transform as trans
import numpy as np

from jittor.dataset.cifar import CIFAR10
from jittor.models.resnet import Resnet50, ResNet
from jittor.models import AlexNet, vgg16
from dataset import Tiny_vid
import pdb
from argparse import ArgumentParser
from bbox import box_iou_batch
# from colorama import Fore, Back, Style


jt.misc.set_global_seed(425)

class DetNet(nn.Module):
    def __init__(self):
        super(DetNet, self).__init__()
        self.preprocess = nn.Sequential(
            nn.BatchNorm2d(3)
        )
        self.backbone = Resnet50(num_classes=5, pretrained=True)
        self.class_head = nn.Sequential(
            nn.Dropout(),
            nn.Linear(5,5),
            # nn.Softmax()
        )
        self.bbox_head = nn.Sequential(
            nn.Linear(5,4)
        )
        
        
    def execute(self, x):
        y = self.preprocess(x)
        # with jt.no_grad():
        y = self.backbone(y)
        class_score = self.class_head(y)
        bbox_out = self.bbox_head(y)
        return class_score, bbox_out



def train(model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        outputs, outboxes = model(inputs)
        with jt.no_grad():
            batch_size = inputs.shape[0]
            pred,_ = jt.argmax(outputs.data, dim=1)
            class_acc = jt.sum(targets[0] == pred) / batch_size
        loss = nn.cross_entropy_loss(outputs, targets[0])
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Acc: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0], class_acc))



def train_class_head(model, train_loader, optimizer, epoch):
    model.train()
    # freeze bbox head
    for name, param in model.class_head.named_parameters():
        param.requires_grad = True
    for name, param in model.backbone.named_parameters():
        param.requires_grad = True
    for name, param in model.preprocess.named_parameters():
        param.requires_grad = True
    for name, param in model.bbox_head.named_parameters():
        param.requires_grad = False
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # import pdb;pdb.set_trace()
        outputs, outboxes = model(inputs)
        with jt.no_grad():
            batch_size = inputs.shape[0]
            pred,_ = jt.argmax(outputs.data, dim=1)
            class_acc = jt.sum(targets[0] == pred) / batch_size
        loss = nn.cross_entropy_loss(outputs, targets[0])
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tTrain Acc: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0], class_acc))


def train_bbox_head(model, train_loader, optimizer, epoch):
    model.train()
    # freeze preprocess and class head
    for name, param in model.class_head.named_parameters():
        param.requires_grad = False
    for name, param in model.backbone.named_parameters():
        param.requires_grad = False
    for name, param in model.preprocess.named_parameters():
        param.requires_grad = False
    for name, param in model.bbox_head.named_parameters():
        param.requires_grad = True
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # pdb.set_trace()
        batch_size = inputs.shape[0]
        outputs, outboxes = model(inputs)
        loss = nn.mse_loss(outboxes, targets[1])
        iou = box_iou_batch(outboxes, targets[1])
        optimizer.step (loss)
        if batch_idx % 10 == 0:
            print(' BOX Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBox_Acc: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader),
                    100. * batch_idx / len(train_loader), loss.data[0], jt.sum(iou > 0.5)/batch_size))



total_acc = 0
total_num = 0
def test(model, val_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs, outboxes = model(inputs)
        pred,_ = jt.argmax(outputs.data, dim=1)
        # pdb.set_trace()
        class_acc = jt.sum(targets[0] == pred)
        test_iou = box_iou_batch(outboxes, targets[1])
        global total_acc
        global total_num 
        total_acc += class_acc
        total_num += batch_size
        # class_acc = class_acc / batch_size
        
    print('Test Epoch: {} [{}/{} ({:.0f}%)]\tAcc: {:.6f}\tTotal Acc: {:.6f}\tBox_Acc: {:.6f}'.format(epoch, \
                batch_idx, len(val_loader),100. * float(batch_idx) / len(val_loader), class_acc/ batch_size, total_acc/total_num, jt.sum(test_iou > 0.5)/batch_size))
    return class_acc    
    # print ('Total test acc =', total_acc / total_num)


def param_dict():
    parser = ArgumentParser(description="Hyper parameters")
    parser.add_argument('-b','--batch_size', default=16, type=int)
    parser.add_argument('-l','--learning_rate', default=1e-2, type=float)
    parser.add_argument('-e','--epochs', default=200, type=int)
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
    
    train_loader = Tiny_vid(train=True, transform=my_transform).set_attrs(batch_size=batch_size, shuffle=True)
    val_loader = Tiny_vid(train=False, transform=my_transform)
    # pdb.set_trace()
    val_loader.set_attrs(batch_size=len(val_loader.ground_truth), shuffle=False)
    

    model = DetNet()
    # model = vgg16(num_classes=5, pretrained=True)
    # optimizer = nn.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # pdb.set_trace()    
    optimizer = nn.SGD(list(filter(lambda val: val.requires_grad, model.parameters())), learning_rate, momentum, weight_decay)
    optimizer_box = nn.SGD(list(filter(lambda val: val.requires_grad, model.parameters())), learning_rate, momentum, weight_decay)
    for epoch in range(epochs):
        train_class_head(model, train_loader, optimizer, epoch)
        train_bbox_head(model, train_loader, optimizer_box, epoch)
        test(model, val_loader, epoch)
        
    

   
        

if __name__=='__main__':
    jt.flags.use_cuda = 1
    prms = param_dict()
    # import pdb; pdb.set_trace()
    main(prms)
    