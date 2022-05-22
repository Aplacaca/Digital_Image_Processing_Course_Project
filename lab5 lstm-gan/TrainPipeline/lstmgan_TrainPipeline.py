import os
import numpy as np
from tqdm import tqdm
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from TrainPipeline.dataset import Weather_Dataset
from utils.visualize import Visualizer
from utils.exception_handler import exception_handler
# from models.dcgan import Generator, Discriminator
from models.wgan_gp import Generator, Discriminator
from models.backbone import FeatureExtractor
from models.conv_lstm import ConvLSTM
from TrainPipeline.dcgan_TrainPipeline import denormalize
import pdb


def recover_img(imgs, img_class='radar'):
    """将图片还原到原始范围"""

    type_id = ['precip', 'radar', 'wind'].index(img_class.lower())
    factor = [10., 70., 35.][type_id]
    imgs = torch.clamp(input=imgs, min=0, max=factor) / factor * 255.0

    return imgs


def compute_gradient_penalty(D, real_samples, fake_samples, Tensor):
    """Calculates the gradient penalty loss for WGAN GP"""

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))

    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha)
                    * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(
        1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


@exception_handler
def lstmgan_TrainPipeline(opt):

    print('LSTM! 🎉🎉🎉')

    # mkdir
    os.makedirs(opt.result_dir, exist_ok=True)
    os.makedirs(opt.result_dir + opt.img_class + '/', exist_ok=True)
    os.makedirs(opt.save_model_file, exist_ok=True)
    os.makedirs(opt.save_model_file + opt.img_class + '/', exist_ok=True)

    feature_extractor = FeatureExtractor(opt.img_size, opt.latent_dim)
    generator = Generator(opt, [1, opt.img_size, opt.img_size])
    discriminator = Discriminator([1, opt.img_size, opt.img_size])
    feature_extractor.load_state_dict(torch.load(
        "./checkpoints/wgan/Radar/fe_9_23000.pth"))
    generator.load_state_dict(torch.load(
        "./checkpoints/wgan/Radar/generator_9_23000.pth"))
    discriminator.load_state_dict(torch.load(
        "./checkpoints/wgan/Radar/discriminator_9_23000.pth"))

    # Loss function
    pred_loss = torch.nn.MSELoss()
    # Loss function
    adversarial_loss = torch.nn.BCELoss()
    # Initialize predictor
    # predictor = ConvLSTM(opt,input_channels=3, hidden_channels=[32, 32], kernel_size=3, step=1,
    #                     effective_step=[0],device = opt.device)
    predictor = torch.nn.LSTM(
        input_size=100, hidden_size=100, batch_first=True, num_layers=5)

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

    optimizer_fe = torch.optim.SGD(feature_extractor.parameters(), lr=opt.lr)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr_g)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr_d)
    optimizer_P = torch.optim.RMSprop(predictor.parameters(), lr=opt.lr_d)

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
        vis = Visualizer(opt.vis_env)

    # ----------
    #  Training
    # ----------

    lambda_gp = 10
    bar_format = '{desc}{n_fmt:>3s}/{total_fmt:<5s} |{bar}|{postfix}'
    print('🚀 开始训练！')

    for epoch in range(opt.n_epochs):
        with tqdm(total=len(dataloader2), bar_format=bar_format) as bar:
            for i, imgs in enumerate(dataloader2):

                bar.set_description(f"\33[36m🌌 Epoch {epoch:1d}")
                # prediction ground truths
                pred_gt = Variable(imgs[20:40, :, :].type(
                    Tensor), requires_grad=False)

                # Configure input
                pred_in = Variable(imgs[0:20, :, :].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(pred_in.shape[0], 1).fill_(
                    1.0), requires_grad=False).to(opt.device)
                fake = Variable(Tensor(pred_in.shape[0], 1).fill_(
                    0.0), requires_grad=False).to(opt.device)

                # Configure input

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # -----------------
                #  Train predictor
                # -----------------

                fe_out = feature_extractor(pred_in).unsqueeze(0)
                # Predict a batch of images
                fe_pred_out, _ = predictor(fe_out)
                fe_pred_out = fe_pred_out.squeeze(dim=0)

                pred_out = generator(fe_pred_out)

                # Real images
                real_validity = discriminator(pred_gt)
                # Fake images
                fake_validity = discriminator(pred_out.detach())
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(
                    discriminator, pred_gt.data, pred_out.detach().data, Tensor)

                d_loss = -torch.mean(real_validity) + \
                    torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_loss.backward()
                optimizer_D.step()

                # -----------------
                #  Train Generator and Feature Extractor predictor every n_critic steps
                # -----------------
                optimizer_G.zero_grad()
                optimizer_fe.zero_grad()
                optimizer_P.zero_grad()

                if i % opt.n_critic == 0:
                    # Loss measures generator's ability to fool the discriminator

                    fake_validity = discriminator(pred_out)
                    g_loss = -torch.mean(fake_validity)
                    g_loss.backward(retain_graph=True)
                    optimizer_G.step()
                    optimizer_fe.step()
                    optimizer_P.step()

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
                    imgs_ = denormalize(imgs.data[:1])
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