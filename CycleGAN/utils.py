import random
import time
import datetime
import sys
import os
from PIL import Image
import pandas as pd
import csv

import torch
from torch.autograd import Variable
import numpy as np


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class Logger():
    def __init__(self, n_epochs, batches_epoch, save_freq,  image_dir='images', log_file='training_log.csv'):
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.epoch_loss = {}  # 当前epoch的损失总和
        self.all_epoch_loss = []  # 所有epoch的平均损失
        self.log_file = log_file
        self.image_dir = image_dir
        self.save_freq = save_freq
        # 创建图像保存目录
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write('\rEpoch %03d/%03d [%04d/%04d] -- ' % (
            self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        # 更新损失
        if losses is not None:
            for loss_name, loss_value in losses.items():
                loss_val = loss_value.item()
                if loss_name not in self.epoch_loss:
                    self.epoch_loss[loss_name] = loss_val
                else:
                    self.epoch_loss[loss_name] += loss_val

            # 输出当前平均损失
            for i, (loss_name, loss_total) in enumerate(self.epoch_loss.items()):
                avg_loss = loss_total / self.batch
                if (i + 1) == len(self.epoch_loss):
                    sys.stdout.write('%s: %.4f -- ' % (loss_name, avg_loss))
                else:
                    sys.stdout.write('%s: %.4f | ' % (loss_name, avg_loss))

        # 计算剩余时间
        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        eta = datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)
        sys.stdout.write('ETA: %s' % str(eta))


        # 检查是否需要保存图像
        if self.epoch % self.save_freq == 0 and images is not None:
            for image_name, tensor in images.items():
                # 转换为图像数组
                image = tensor2image(tensor.data)
                image = image.transpose((1, 2, 0))
                # 生成文件名，使用epoch号，可能不需要batch号？
                # 或者保留batch号，因为此时是最后一个batch
                filename = f"{image_name}_epoch{self.epoch}_batch{self.batch}.png"
                save_path = os.path.join(self.image_dir, filename)
                img = Image.fromarray(image)
                img.save(save_path)

        # 检查是否是当前epoch的最后一批
        if self.batch % self.batches_epoch == 0:
            # 计算当前epoch的平均损失
            current_epoch_loss = {
                loss_name: total / self.batch
                for loss_name, total in self.epoch_loss.items()
            }
            self.all_epoch_loss.append({
                'epoch': self.epoch,
                **current_epoch_loss
            })

            # 保存损失到CSV
            self._save_log_to_csv()

            # 重置计数器
            self.epoch += 1
            self.batch = 1
            self.epoch_loss = {}
            sys.stdout.write('\n')
        else:
            self.batch += 1

    def _save_log_to_csv(self):
        # 保存当前epoch的损失数据到CSV
        if not self.all_epoch_loss:
            return

        with open(self.log_file, 'a', newline='') as f:
            # 写入表头
            if f.tell() == 0:
                fieldnames = ['epoch'] + list(self.all_epoch_loss[-1].keys())[1:]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

            # 写入数据
            writer = csv.DictWriter(f, fieldnames=self.all_epoch_loss[-1].keys())
            for data in self.all_epoch_loss:
                writer.writerow(data)

        # 清空缓冲区
        self.all_epoch_loss = []


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), "缓冲区大小必须大于0"
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.random() > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "衰减必须在训练开始前"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)