import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tcn import TemporalConvNet

# 分类器 C
class Classifier(nn.Module):
    def __init__(self, in_channel_num, f_c_dim=24, category_num=10):
        super(Classifier, self).__init__()

        self.tcn1 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[64] * 6,
            kernel_size=3,
            dropout=0.2
        )

        self.tcn2 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[32] * 3,
            kernel_size=5,
            dropout=0.2
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 + 32, f_c_dim),
            nn.LeakyReLU(0.2, True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(f_c_dim, category_num),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, C, T]
        tcn1 = self.tcn1(data)[:, :, -1] # [N, C]
        tcn2 = self.tcn2(data)[:, :, -1]

        x = torch.cat((tcn1, tcn2), dim=1)

        f_c = self.fc1(x)
        x = self.fc2(f_c) # one-hot [N, category_num]
        return f_c, x

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channel_num, z_dim=64):
        super(Encoder, self).__init__()

        self.tcn1 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[64] * 6,
            kernel_size=3,
            dropout=0.2
        )

        self.tcn2 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[32] * 3,
            kernel_size=5,
            dropout=0.2
        )

        self.tcn3 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[16] * 2,
            kernel_size=7,
            dropout=0.2
        )

        self.fc1 = nn.Linear(64 + 32 + 16, z_dim)
        self.fc2 = nn.Linear(64 + 32 + 16, z_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(mean)
        return mean + eps * std

    def forward(self, data):
        tcn1 = self.tcn1(data)[:, :, -1] # [N, C]
        tcn2 = self.tcn2(data)[:, :, -1]
        tcn3 = self.tcn3(data)[:, :, -1]

        x = torch.cat((tcn1, tcn2, tcn3), dim=1)

        mean = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar


class ResBlock(nn.Module):
  def __init__(self, c_in_dim, c_out_dim, t_dim):
    super(ResBlock, self).__init__()
    self.conv = nn.Sequential(
        nn.LayerNorm((c_in_dim, t_dim)),
        nn.LeakyReLU(0.2, True),
        nn.Conv1d(c_in_dim, c_in_dim, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm((c_in_dim, t_dim)),
        nn.LeakyReLU(0.2, True),
        nn.Conv1d(c_in_dim, c_out_dim, kernel_size=3, stride=1, padding=1),
        nn.Upsample(scale_factor=2) # t_dim * 2
    )

    self.short_cut = nn.Sequential(
      nn.Upsample(scale_factor=2),
      nn.Conv1d(c_in_dim, c_out_dim, kernel_size=1, stride=1, padding=0)
    )

  def forward(self, x):
    out = self.conv(x) + self.short_cut(x)
    return out

# 生成 [96, 240]
class Generator(nn.Module):
    def __init__(self, z_dim=32, c_dim=32, out_size=(96, 240)):
        super(Generator, self).__init__()
        self.out_size = out_size
        # [96, 240]
        self.fc = nn.Linear(z_dim + c_dim, 512)

        self.up = nn.Upsample(scale_factor=4)
        self.block1 = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block2 = ResBlock(256, 256, 4)
        self.block3 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block4 = ResBlock(256, 256, 8)
        self.block5 = ResBlock(256, 256, 16)
        self.block6 = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, True)
        )
        self.block7 = ResBlock(128, 128, 32)
        self.block8 = ResBlock(128, 96, 64)
        self.block9 = ResBlock(96, 96, 128)
        self.block10 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        x = self.fc(x) # [N, 512]
        x = x.view(x.size(0), -1, 1) # [N, 512, 1]
        x = self.up(x) # [N, 512, 4]
        x = self.block1(x) # [N, 256, 4]
        x = self.block2(x) # [N, 256, 8]
        x = self.block3(x) # [N, 256, 8]
        x = self.block4(x) # [N, 256, 16]
        x = self.block5(x) # [N, 256, 32]
        x = self.block6(x) # [N, 128, 32]
        x = self.block7(x) # [N, 128, 64]
        x = self.block8(x) # [N, 96, 128]
        x = self.block9(x) # [N, 96, 256]
        x = self.block10(x) # [N, 96, 256]
        x = x[:, :self.out_size[0], :self.out_size[1]] # [N, 96, 240]
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
    def __init__(self, in_channel_num, f_d_dim=64):
        super(Discriminator, self).__init__()

        self.tcn1 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[64] * 6,
            kernel_size=3,
            dropout=0.2
        )

        self.tcn2 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[32] * 3,
            kernel_size=5,
            dropout=0.2
        )

        self.tcn3 = TemporalConvNet(
            num_inputs=in_channel_num,
            num_channels=[16] * 2,
            kernel_size=7,
            dropout=0.2
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64 + 32 + 16, f_d_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(f_d_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, 96, 239]
        tcn1 = self.tcn1(data)[:, :, -1] # [N, C]
        tcn2 = self.tcn2(data)[:, :, -1] # [N, C]
        tcn3 = self.tcn3(data)[:, :, -1] # [N, C]

        x = torch.cat((tcn1, tcn2, tcn3), dim=1)

        f_d = self.fc1(x)
        x = self.fc2(f_d)
        x = x.squeeze(1) # [N]
        return f_d, x
