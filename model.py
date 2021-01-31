import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tcn import TemporalConvNet

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm1d(in_dim),
            nn.LeakyReLU(0.2, True),
            nn.Conv1d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2) # t_dim * 2
        )

        self.short_cut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv1d(in_dim, out_dim, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out

# 分类器 C
class Classifier(nn.Module):
    def __init__(self, in_channel_num, f_c_dim=24, category_num=10):
        '''
        in_channel_num: 单样本通道数，[样本个数, 单样本输入的通道数, 样本序列长度]
        f_c_dim: 输出的潜在向量 identity 的长度
        category_num: 分类 one-hot 向量长度
        '''
        super(Classifier, self).__init__()

        channels = [64] * 8 # [第1层卷积核数量，第2层卷积核数量, ..., 第levels层卷积核数量]
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=3,
            dropout=0.2
        )

        self.fc1 = nn.Sequential(
            nn.Linear(channels[-1], f_c_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(f_c_dim, category_num),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, C, T]
        x = self.tcn(data) # [N, C, T]
        x = x[:, :, -1] # [N, 128]

        f_c = self.fc1(x)
        x = self.fc2(f_c) # one-hot [N, category_num]
        return f_c, x

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channel_num, z_dim=64):
        super(Encoder, self).__init__()

        channels = [64] * 8
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=3,
            dropout=0.2
        )
        self.fc1 = nn.Linear(channels[-1], z_dim)
        self.fc2 = nn.Linear(channels[-1], z_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(mean)
        return mean + eps * std

    def forward(self, data):
        x = self.tcn(data) # [N, C, T]
        x = x[:, :, -1] # [N, C]
        mean = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

# 生成 [239, 96]
class Generator(nn.Module):
    def __init__(self, z_dim=256, c_dim=256, out_size=(96, 239)):
        super(Generator, self).__init__()
        self.out_size = out_size

        self.fc = nn.Linear(z_dim + c_dim, 512)
        self.up = nn.Upsample(scale_factor=8)
        self.block1 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block2 = ResBlock(256, 256)
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block4 = ResBlock(256, 256)
        self.block5 = ResBlock(256, 256)
        self.block6 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block7 = ResBlock(128, 128)
        self.block8 = ResBlock(128, 96)
        self.block9 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1) # [n, 512, 1]
        x = self.up(x) # [n, 512, 8]
        x = self.block1(x) # [n, 256, 8]
        x = self.block2(x) # [n, 256, 16]
        x = self.block3(x) # [n, 256, 16]
        x = self.block4(x) # [n, 256, 32]
        x = self.block5(x) # [n, 256, 64]
        x = self.block6(x) # [n, 128, 64]
        x = self.block7(x) # [n, 128, 128]
        x = self.block8(x) # [n, 96, 256]
        x = self.block9(x) # [n, 96, 256]
        x = x[:, :self.out_size[0], 8:8+self.out_size[1]] # 实际只使用 [n, 239, 96]
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channel_num, f_d_dim=64):
        super(Discriminator, self).__init__()

        channels = [64] * 8
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=3,
            dropout=0.2
        )

        self.fc1 = nn.Sequential(
            nn.Linear(channels[-1], f_d_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(f_d_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, 96, 239]
        x = self.tcn(data) # [N, C, T]
        x = x[:, :, -1] # [N, C]

        f_d = self.fc1(x)
        x = self.fc2(f_d)
        x = x.squeeze(1) # [N]
        return f_d, x
