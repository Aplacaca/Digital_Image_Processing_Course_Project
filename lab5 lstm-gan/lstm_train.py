import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from config import DefaultConfig, TSConfig
from dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.setup_seed import setup_seed
from utils.exception_handler import exception_handler
from models.dcgan import Generator as dc_generator, Discriminator as dc_disciminator
from models.conv_lstm import ConvLSTM
import pdb
# config
opt = TSConfig()

# global config
setup_seed(opt.seed)
torch.cuda.set_device(1)

# mkdir
os.makedirs(opt.result_dir, exist_ok=True)
os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
os.makedirs(opt.save_model_file, exist_ok=True)
os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)


def recover_img(imgs, img_class=opt.img_class):
    """将图片还原到原始范围"""

    type_id = ['precip', 'radar', 'wind'].index(img_class.lower())
    factor = [10., 70., 35.][type_id]
    imgs = torch.clamp(input=imgs, min=0, max=factor) / factor * 255

    return imgs


@exception_handler
def train():
    # Loss function
    pred_loss = torch.nn.MSELoss()

    # Initialize predictor
    predictor = ConvLSTM(opt,input_channels=1, hidden_channels=[32, 32], kernel_size=3, step=20,
                        effective_step=list(range(0,20)))

    if opt.use_gpu:
        predictor.to(opt.device)
        pred_loss.to(opt.device)

    # Optimizers
    optimizer_TS = torch.optim.Adam(
        predictor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets = Weather_Dataset(img_dir=opt.train_dataset_path + opt.img_class,
                               csv_path=opt.train_csv_path,
                               img_size=opt.img_size)
    dataloader = iter(range(len(datasets)))
    # dataloader = DataLoader(datasets,batch_size = 16, shuffle=False)
    # start visualization
    if opt.vis:
        vis = Visualizer(opt.vis_env,port=8099)

    # ----------
    #  Training
    # ----------

    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(datasets), bar_format=bar_format) as bar:
            for i, imgs_index in enumerate(dataloader):
                imgs = datasets[imgs_index]
            # for imgs in dataloader:
                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # prediction ground truths
                pred_gt = Variable(imgs[20:40,:,:,:].type(Tensor), requires_grad=False)
    
                # Configure input
                pred_in = Variable(imgs[0:20,:,:,:].type(Tensor))

                # -----------------
                #  Train predictor
                # -----------------

                predictor.zero_grad()

                # Predict a batch of images
                pred_out = predictor(pred_in)

                # Loss measures generator's ability to fool the discriminator
                ts_loss = pred_loss(pred_out, pred_gt)

                ts_loss.backward()
                optimizer_TS.step()


                # display the last part of progress bar
                bar.set_postfix_str(
                    f'TS loss: {ts_loss.item():.3f}\33[0m')
                bar.update()

                # visualize the loss curve and generated images in visdom
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='TS loss', y=ts_loss.item())
                if opt.vis:
                    imgs_ = recover_img(imgs.data[:1], opt.img_class)
                    pred_imgs_ = recover_img(pred_out.data[:1], opt.img_class)
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=pred_imgs_, nrow=1)

                # save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
                    pred_imgs_ = recover_img(pred_out.data[:9], opt.img_class)
                    save_image(pred_imgs_, opt.result_dir + opt.img_class +
                               '/' + f"{i}.png", nrow=3, normalize=False)
                    torch.save(predictor.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + 'preditor_'+str(i)+'.pth')


if __name__ == '__main__':
    train()
