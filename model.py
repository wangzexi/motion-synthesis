import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from tcn import TemporalConvNet


# 这是生成器使用的放大的残差块
# 可能得改成膨胀卷积的那种
class ResBlock(nn.Module):
    """
    This residual block is different with the one we usually know which consists of
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    Also spatial size is decreased by half because of AvgPool2d.
    """

    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

        self.short_cut = nn.Sequential(nn.Upsample(scale_factor=2),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out

# 这是编码器使用的缩小的残差块
class ResidualBlock(nn.Module):
    """
    This residual block is different with the one we usually know which consists of
    [conv - norm - act - conv - norm] and identity mapping(x -> x) for shortcut.
    Also spatial size is decreased by half because of AvgPool2d.
    """

    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(0.2, True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


# 分类器 C
class Classifier(nn.Module):
    def __init__(self, in_channel_num, f_c_dim=24, category_num=10, kernel_size=3, dropout=0.2):
        '''
        in_channel_num: 单样本通道数，[样本个数, 单样本输入的通道数, 样本序列长度]
        identity_dim：输出的潜在向量 identity 的长度
        category_num：分类 one-hot 向量长度
        level_channel_num: 每层卷积核的数量，最终输出的通道数
        level_num: 膨胀卷积层数，层数高则感知野大，某层感知野的范围是2的幂
        kernel_size: 卷积核大小
        dropout: 暂时不参与响应神经元比率
        '''
        super(Classifier, self).__init__()

        channels = [64] * 8 # [第1层卷积核数量，第2层卷积核数量, ..., 第levels层卷积核数量]
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout
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
    def __init__(self, in_channel_num, z_dim=64, kernel_size=3, dropout=0.2):
        super(Encoder, self).__init__()

        channels = [64] * 8
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout
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
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block2 = ResBlock(256, 256)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block4 = ResBlock(256, 256)
        self.block5 = ResBlock(256, 256)
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block7 = ResBlock(128, 128)
        self.block8 = ResBlock(128, 64)
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1) # [n, 512, 1, 1]
        x = self.up(x) # [n, 512, 8, 8]
        x = self.block1(x) # [n, 256, 8, 8]
        x = self.block2(x) # [n, 256, 16, 16]
        x = self.block3(x) # [n, 256, 16, 16]
        x = self.block4(x) # [n, 256, 32, 32]
        x = self.block5(x) # [n, 256, 64, 64]
        x = self.block6(x) # [n, 128, 64, 64]
        x = self.block7(x) # [n, 128, 128, 128]
        x = self.block8(x) # [n, 64, 256, 256]
        x = self.block9(x) # [n, 1, 256, 256]
        x = x[:, :, :self.out_size[0], :self.out_size[1]] # 实际只使用 [n, 1, 239, 96]
        x = x.view(x.size(0), x.size(2), x.size(3)) # [n, 239, 96]
        return x


# 使用 TCN 生成 [239, 96]
class GeneratorTCN(nn.Module):
    def __init__(self, identity_dim=256, attribute_dim=256, out_size=(96, 239), kernel_size=3, dropout=0.2):
        super(GeneratorTCN, self).__init__()
        self.out_size = out_size
        
        self.fc = nn.Linear(identity_dim + attribute_dim, out_size[1])
        channels = [out_size[0]] * 8
        self.tcn = TemporalConvNet(
            num_inputs=out_size[0],
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.tanh = nn.Tanh()

    def forward(self, identity, attribute):
        x = torch.cat((identity, attribute), 1)

        x = self.fc(x) # [N, T]
        x = x.view(x.size(0), self.out_size[0], -1) # [N, out_size[0], T]
        print(x.shape)
        exit()
        x = self.tcn(x) # [N, C, T]
        x = self.tanh(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channel_num, f_d_dim=64, kernel_size=2, dropout=0.2):
        super(Discriminator, self).__init__()

        channels = [64] * 8
        self.tcn = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=channels,
            kernel_size=kernel_size,
            dropout=dropout
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
