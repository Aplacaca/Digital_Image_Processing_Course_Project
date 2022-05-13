# coding:utf-8
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets
from torch.autograd import Variable
from torchvision.utils import save_image

from config import DefaultConfig
from utils.setup_seed import setup_seed
from utils.visualize import Visualizer
from dataset import Weather_Dataset

# config
opt = DefaultConfig()

setup_seed(opt.seed)
torch.cuda.set_device(1)
os.makedirs(opt.result_dir, exist_ok=True)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(
            nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(
                0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        # Initialize weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if opt.use_gpu:
    generator.to(opt.device)
    discriminator.to(opt.device)
    adversarial_loss.to(opt.device)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

# Configure data loader
datasets = Weather_Dataset(img_dir='./weather_data/Train/Precip',
                           csv_path='./weather_data/dataset_train.csv',
                           img_size=opt.img_size)
dataloader = iter(range(len(datasets)))

# start visualization
if opt.vis:
    vis = Visualizer(opt.vis_env)

# ----------
#  Training
# ----------

bar_format = '{desc}{n_fmt:>4s}/{total_fmt:<4s}|{bar}|{postfix}'
print('🚀 开始训练！')


for epoch in range(opt.n_epochs):
    with tqdm(total=len(datasets), bar_format=bar_format) as bar:
        for i, imgs_index in enumerate(dataloader):
            # 刷新进度条前半部分
            bar.set_description(f"\33[36m🌌 Epoch {epoch:2d}")

            imgs = datasets[imgs_index]

            # Adversarial ground truths
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(
                1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(
                0.0), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(
                0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(
                discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            # 刷新进度条后半部分
            bar.set_postfix_str(
                f'D loss: {d_loss.item():.3f}, G loss: {g_loss.item():.3f}\33[0m')
            bar.update()

            # 实时显示loss和效果图
            if opt.vis and i % 30 == 0:
                vis.plot(win='Loss', name='G loss', y=g_loss.item())
                vis.plot(win='Loss', name='D loss', y=d_loss.item())
            if opt.vis:
                vis.img(name='Real', img_=imgs.data[:1], nrow=1)
                vis.img(name='Fake', img_=gen_imgs.data[:1], nrow=1)

            # 定期保存效果图
            batches_done = epoch * len(datasets) + i
            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:25], "images_dcgan/%d.png" %
                           batches_done, nrow=5, normalize=True)
