#!/usr/bin/python3

import argparse
import sys
import os
from pathlib import Path
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import torch
from PIL import Image
from models import Generator
from datasets import ImageDataset


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='批次大小')
    parser.add_argument('--dataroot', type=str, default='datasets/wheat/', help='数据集根目录')
    parser.add_argument('--input_nc', type=int, default=3, help='输入通道数')
    parser.add_argument('--output_nc', type=int, default=3, help='输出通道数')
    parser.add_argument('--size', type=int, default=256, help='数据缩放尺寸（平方）')
    parser.add_argument('--cuda', default=True, help='是否使用GPU')
    parser.add_argument('--n_cpu', type=int, default=8, help='数据加载线程数')
    parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A→B生成器权重路径')
    parser.add_argument('--generator_B2A', type=str, default='output/netG_B2A.pth', help='B→A生成器权重路径')
    parser.add_argument('--output_dir', type=str, default='fake/train', help='输出结果保存目录')
    parser.add_argument('--data', type=str, default='train', help='生成训练集或测试集')
    return parser.parse_args()


def setup_devices(opt):
    """设置设备（CPU/GPU）"""
    return torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")


def load_models(opt, device):
    """加载并配置生成器网络"""
    netG_A2B = Generator(opt.input_nc, opt.output_nc).to(device)
    netG_B2A = Generator(opt.output_nc, opt.input_nc).to(device)

    netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    netG_B2A.load_state_dict(torch.load(opt.generator_B2A))

    netG_A2B.eval()
    netG_B2A.eval()

    return netG_A2B, netG_B2A


def create_dataloader(opt):
    """创建数据加载器"""
    transform_ops = [
        transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
        transforms.CenterCrop(opt.size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    dataset = ImageDataset(opt.dataroot, transforms_=transform_ops, mode=opt.data)
    return DataLoader(dataset,
                      batch_size=opt.batchSize,
                      shuffle=False,
                      num_workers=opt.n_cpu)


def prepare_output_directories(opt):
    """创建输出目录结构"""
    output_root = Path(opt.output_dir)
    output_A_dir = output_root / 'A'
    output_B_dir = output_root / 'B'

    output_A_dir.mkdir(parents=True, exist_ok=True)
    output_B_dir.mkdir(parents=True, exist_ok=True)

    return output_A_dir, output_B_dir


def test_cycle_gan(opt):
    """主测试流程"""
    device = setup_devices(opt)
    netG_A2B, netG_B2A = load_models(opt, device)
    dataloader = create_dataloader(opt)
    output_A_dir, output_B_dir = prepare_output_directories(opt)

    with torch.no_grad():
        for i, batch in enumerate(dataloader, 1):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)

            # 生成对抗转换
            fake_B = 0.5 * (netG_A2B(real_A).data + 1.0)
            fake_A = 0.5 * (netG_B2A(real_B).data + 1.0)

            # 保存结果
            save_image(fake_A, str(output_A_dir / f"{i:04d}.png"))
            save_image(fake_B, str(output_B_dir / f"{i:04d}.png"))

            sys.stdout.write(f'\r生成图片 {i}/{len(dataloader)}')

    print("\n测试完成！结果保存在:", opt.output_dir)


if __name__ == "__main__":
    opt = parse_arguments()
    print(f"当前配置：\n{opt}")
    test_cycle_gan(opt)
