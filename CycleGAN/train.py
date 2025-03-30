#!/usr/bin/python3

import argparse
import itertools
import os

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
from utils import Logger
from utils import weights_init_normal
from datasets import ImageDataset



if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='datasets/wheat/', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=100,
                        help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', default=True, help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_freq', type=int, default=20, help='训练过程图片保存间隔批次')
    opt = parser.parse_args()
    print(opt)
    # 创建输出目录
    os.makedirs('output', exist_ok=True)

    # 设备配置
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    ###### 定义变量 ######
    # 网络
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)
    netD_A = Discriminator(opt.input_nc).to(device)
    netD_B = Discriminator(opt.output_nc).to(device)

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    # 损失函数
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # 优化器与学习率调度器
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                   lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G,
                                                       lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                         lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    # 目标张量
    Tensor = torch.cuda.FloatTensor if device.type == 'cuda' else torch.Tensor
    target_real = torch.full((opt.batchSize,), 1.0, device=device, requires_grad=False)
    target_fake = torch.full((opt.batchSize,), 0.0, device=device, requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # 数据集加载器
    transforms_ = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
                   transforms.RandomCrop(opt.size),
                   transforms.RandomHorizontalFlip(),
                   transforms.ToTensor(),
                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    dataloader = DataLoader(
        ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=opt.n_cpu
    )

    # 训练日志
    logger = Logger(opt.n_epochs, len(dataloader), opt.save_freq)
    ###################################

    ###### 训练循环 ######
    for epoch in range(opt.epoch, opt.n_epochs):
        for i, batch in enumerate(dataloader):
            # 数据加载到设备
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            ###### 生成器A2B和B2A ######
            optimizer_G.zero_grad()

            # 身份损失
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0

            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN损失
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # 循环一致性损失
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # 总损失
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### 判别器A ######
            optimizer_D_A.zero_grad()

            # 真实样本损失
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # 伪造样本损失（使用缓冲区）
            fake_A = fake_A_buffer.push_and_pop(fake_A.detach())
            pred_fake = netD_A(fake_A)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### 判别器B ######
            optimizer_D_B.zero_grad()

            # 真实样本损失
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # 伪造样本损失（使用缓冲区）
            fake_B = fake_B_buffer.push_and_pop(fake_B.detach())
            pred_fake = netD_B(fake_B)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            # 日志记录
            logger.log({
                'loss_G': loss_G,
                'loss_G_identity': (loss_identity_A + loss_identity_B),
                'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
                'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB),
                'loss_D': (loss_D_A + loss_D_B),
            },
                images={
                    'real_A': real_A,
                    'real_B': real_B,
                    'fake_A': fake_A,
                    'fake_B': fake_B
                })

        # 学习率更新
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        # 保存模型
        torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
        torch.save(netG_B2A.state_dict(), 'output/netG_B2A.pth')
        torch.save(netD_A.state_dict(), 'output/netD_A.pth')
        torch.save(netD_B.state_dict(), 'output/netD_B.pth')
    ###################################