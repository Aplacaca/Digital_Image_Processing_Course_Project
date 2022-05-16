import os
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from zmq import device

from config import DefaultConfig, TSConfig
from dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.setup_seed import setup_seed
from utils.exception_handler import exception_handler
from models.dcgan import Generator as dc_generator, Discriminator as dc_disciminator
from models.backbone import FeatureExtractor
from models.conv_lstm import ConvLSTM
from gan_train import denormalize
import pdb
# config
opt = TSConfig()

# global config
setup_seed(opt.seed)
torch.cuda.set_device(0)

# mkdir
os.makedirs(opt.result_dir, exist_ok=True)
os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
os.makedirs(opt.save_model_file, exist_ok=True)
os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)


def recover_img(imgs, img_class=opt.img_class):
    """将图片还原到原始范围"""

    type_id = ['precip', 'radar', 'wind'].index(img_class.lower())
    factor = [10., 70., 35.][type_id]
    imgs = torch.clamp(input=imgs, min=0, max=factor) / factor * 255.0

    return imgs


@exception_handler
def train():
    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = dc_generator(opt)
    discriminator = dc_disciminator(opt)
    feature_extractor.load_state_dict(torch.load("./checkpoints/dcgan/Radar/fe_20000.pth"))
    generator.load_state_dict(torch.load("./checkpoints/dcgan/Radar/generator_20000.pth"))
    discriminator.load_state_dict(torch.load("./checkpoints/dcgan/Radar/discriminator_9000.pth"))
    
    # Loss function
    pred_loss = torch.nn.MSELoss()
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize predictor
    # predictor = ConvLSTM(opt,input_channels=3, hidden_channels=[32, 32], kernel_size=3, step=1,
    #                     effective_step=[0],device = opt.device)
    predictor = torch.nn.LSTM(input_size=100, hidden_size=100, batch_first=True)

    if opt.use_gpu:
        predictor.to(opt.device)
        pred_loss.to(opt.device)
        adversarial_loss.to(opt.device)
        feature_extractor.to(opt.device)
        generator.to(opt.device)
        discriminator.to(opt.device)

    # Optimizers
    optimizer_TS = torch.optim.Adam(
        predictor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    optimizer_fe = torch.optim.Adam(feature_extractor.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    
    Tensor = torch.cuda.FloatTensor if opt.use_gpu else torch.FloatTensor

    # Configure data loader
    datasets1 = Weather_Dataset(img_dir=opt.train_dataset_path + 'Precip',
                               csv_path=opt.train_csv_path,
                               img_size=opt.img_size)
    dataloader1 = iter(range(len(datasets1)))
    datasets2 = Weather_Dataset(img_dir=opt.train_dataset_path + 'Radar',
                               csv_path=opt.train_csv_path,
                               img_size=opt.img_size)
    dataloader2 = iter(range(len(datasets2)))
    datasets3 = Weather_Dataset(img_dir=opt.train_dataset_path + 'Wind',
                               csv_path=opt.train_csv_path,
                               img_size=opt.img_size)
    dataloader3 = iter(range(len(datasets3)))
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
        with tqdm(total=len(datasets1), bar_format=bar_format) as bar:
            for i, imgs_index in enumerate(dataloader1):
                imgs1 = datasets1[imgs_index] # precip
                imgs2 = datasets2[imgs_index] # radar
                imgs3 = datasets3[imgs_index] # wind
                # imgs = torch.cat([imgs1,imgs2,imgs3], dim=1)
            # for imgs in dataloader:
                # display the first part of progress bar
                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")

                # prediction ground truths
                pred_gt = Variable(imgs2[20:40,:,:,:].type(Tensor), requires_grad=False)
    
                # Configure input
                pred_in = Variable(imgs2[0:20,:,:,:].type(Tensor))
                
                fe_out = feature_extractor(pred_in).unsqueeze(0)
                
                
                 # Adversarial ground truths
                valid = Variable(Tensor(pred_in.shape[0], 1).fill_(
                    1.0), requires_grad=False).to(opt.device)
                fake = Variable(Tensor(pred_in.shape[0], 1).fill_(
                    0.0), requires_grad=False).to(opt.device)

                # Configure input
                 # -----------------
                #  Train Generator and Feature Extractor
                # -----------------

                optimizer_G.zero_grad()
                optimizer_fe.zero_grad()

                # -----------------
                #  Train predictor 
                # -----------------


                # Predict a batch of images
                fe_pred_out,_ = predictor(fe_out)
                fe_pred_out = fe_pred_out.squeeze(dim=0)

                pred_out = generator(fe_pred_out)
                # pred_out = generator(fe_out.squeeze(dim=0))
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(pred_out), valid)
                g_loss.backward()
                optimizer_G.step()
                optimizer_fe.step()
                
                
                # pdb.set_trace()


                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(pred_gt), valid)
                fake_loss = adversarial_loss(
                    discriminator(pred_out.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                predictor.zero_grad()
                
                ts_loss = pred_loss(pred_out.detach(), pred_gt)
                # pdb.set_trace()
                # ts_loss.backward()
                optimizer_TS.step()
                
                # display the last part of progress bar
                bar.set_postfix_str(
                    f'TS loss: {ts_loss.item():.3f}\33[0m')
                bar.update()

                # visualize the loss curve and generated images in visdom
                if opt.vis and i % 50 == 0:
                    vis.plot(win='Loss', name='TS loss', y=ts_loss.item())
                if opt.vis:
                    imgs_ = denormalize(imgs2.data[:1])
                    pred_imgs_ = denormalize(pred_out.data[:1])
                    # imgs_ = recover_img(imgs2.data[:1], opt.img_class)
                    # pred_imgs_ = recover_img(pred_out.data[:1], opt.img_class)
                    # pdb.set_trace()
                    vis.img(name='Real', img_=imgs_, nrow=1)
                    vis.img(name='Fake', img_=pred_imgs_, nrow=1)

                # save the model and generated images every 500 batches
                if i % opt.sample_interval == 0:
                    pred_imgs_ = denormalize(pred_out.data[:9])
                    save_image(pred_imgs_, opt.result_dir + opt.img_class +
                               '/' + f"{i}.png", nrow=3, normalize=False)
                    torch.save(predictor.state_dict(),
                               opt.save_model_file + opt.img_class + '/' + 'preditor_'+str(i)+'.pth')


if __name__ == '__main__':
    train()
