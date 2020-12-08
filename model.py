import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
# from net_sphere import sphere20a # 这个东西好像没用到
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
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(in_dim, affine=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
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
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool2d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        out = self.conv(x) + self.short_cut(x)
        return out


## 1个动作动画 = 96关节通道 * 240帧数

# I 与 C 都是这个玩意
class EncoderTCN(nn.Module):
    def __init__(self, input_size, latent_dim=24, category_num=10, level_channel_num=32, level_num=6, kernel_size=2, dropout=0.2):
        '''
        input_size: 单样本通道数，[样本个数, 单样本输入的通道数, 样本序列长度]
        latent_dim：输出的潜在向量 identity 的长度
        category_num：分类 one-hot 向量长度
        level_channel_num: 每层卷积核的数量，最终输出的通道数
        level_num: 膨胀卷积层数，层数高则感知野大，某层感知野的范围是2的幂
        kernel_size: 卷积核大小
        dropout: 暂时不参与响应神经元比率
        '''
        super(EncoderTCN, self).__init__()

        channels = [level_channel_num] * level_num # [第1层卷积核数量，第2层卷积核数量, ..., 第levels层卷积核数量]
        self.tcn = TemporalConvNet(
            input_size,
            channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # TCN最后一层接线性层
        self.fc_ctg = nn.Linear(channels[-1], category_num) # 分类 one-hot 向量
        self.fc_id = nn.Linear(channels[-1], latent_dim) # 压缩出的 identity 向量

    def forward(self, data):
        # inputs: [样本个数, 单样本输入的通道数, 样本序列长度]
        # 单样本输入的通道数为帧数

        # [样本个数, 卷积核数量num_channels, 样本序列长度]
        x = self.tcn(data)

        # [样本个数, 卷积核数量num_channels]
        x = x[:, :, -1] # 对每个输出通道，取出最后一个神经元的值，这些玩意卷积了它们前面的序列元素

        # [样本个数, output_size]
        ctg = self.fc_ctg(x)
        identity = self.fc_id(x)
        return ctg, identity # 分类向量，压缩出的 identity

## 上面的 TCN 取代了这个
# class Encoder(nn.Module):
#     # c_dim 是倒二层输出的「特征向量」
#     def __init__(self, c_dim=347):
#         super(Encoder, self).__init__()

#         self.conv = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
#         self.res_blocks = nn.Sequential(ResidualBlock(64, 128),
#                                         ResidualBlock(128, 192),
#                                         ResidualBlock(192, 256))
#         self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                         nn.AvgPool2d(kernel_size=4, stride=4, padding=0))

#         self.fc = nn.Linear(1024, c_dim)

#     def forward(self, x):
#         out = self.conv(x)
#         out = self.res_blocks(out)
#         out = self.pool_block(out)
#         out = out.view(x.size(0), -1)
#         mu = self.fc(out)

#         return mu, out
###

# 特征提取也用 TCN 吧
class AttributeTCN(nn.Module):
    def __init__(self, input_size, latent_dim=24, category_num=10, level_channel_num=32, level_num=6, kernel_size=2, dropout=0.2):
        super(AttributeTCN, self).__init__()

        channels = [level_channel_num] * level_num
        self.tcn = TemporalConvNet(
            input_size,
            channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc_mu = nn.Linear(channels[-1], latent_dim) # VAE 的 mu
        self.fc_var = nn.Linear(channels[-1], latent_dim) # VAE 的 var

    def forward(self, data):
        x = self.tcn(data)
        x = x[:, :, -1]

        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

# # 换为上面的 TCN 了
# class Attribute(nn.Module):
#     def __init__(self, latent_dim=8, feature=False):
#         super(Attribute, self).__init__()
#         """
#             Output is mu and log(var) for re-parameterization trick used in Variation Auto Encoder.
#             Encoding is done in this order.
#             1. Use this encoder and get mu and log_var
#             2. std = exp(log(var / 2))
#             3. random_z = N(0, 1)
#             4. encoded_z = random_z * std + mu (Re-parameterization trick)
#         """
#         self.feature = feature
#         #input = B*3*112*96
#         self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
#         self.relu1_1 = nn.PReLU(64)
#         self.conv1_2 = nn.Conv2d(64,64,3,1,1)
#         self.relu1_2 = nn.PReLU(64)
#         self.conv1_3 = nn.Conv2d(64,64,3,1,1)
#         self.relu1_3 = nn.PReLU(64)

#         self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
#         self.relu2_1 = nn.PReLU(128)
#         self.conv2_2 = nn.Conv2d(128,128,3,1,1)
#         self.relu2_2 = nn.PReLU(128)
#         self.conv2_3 = nn.Conv2d(128,128,3,1,1)
#         self.relu2_3 = nn.PReLU(128)

#         self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
#         self.relu2_4 = nn.PReLU(128)
#         self.conv2_5 = nn.Conv2d(128,128,3,1,1)
#         self.relu2_5 = nn.PReLU(128)

#         self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
#         self.relu3_1 = nn.PReLU(256)
#         self.conv3_2 = nn.Conv2d(256,256,3,1,1)
#         self.relu3_2 = nn.PReLU(256)
#         self.conv3_3 = nn.Conv2d(256,256,3,1,1)
#         self.relu3_3 = nn.PReLU(256)

#         self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
#         self.relu3_4 = nn.PReLU(256)
#         self.conv3_5 = nn.Conv2d(256,256,3,1,1)
#         self.relu3_5 = nn.PReLU(256)

#         self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
#         self.relu3_6 = nn.PReLU(256)
#         self.conv3_7 = nn.Conv2d(256,256,3,1,1)
#         self.relu3_7 = nn.PReLU(256)

#         self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
#         self.relu3_8 = nn.PReLU(256)
#         self.conv3_9 = nn.Conv2d(256,256,3,1,1)
#         self.relu3_9 = nn.PReLU(256)

#         self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
#         self.relu4_1 = nn.PReLU(512)
#         self.conv4_2 = nn.Conv2d(512,512,3,1,1)
#         self.relu4_2 = nn.PReLU(512)
#         self.conv4_3 = nn.Conv2d(512,512,3,1,1)
#         self.relu4_3 = nn.PReLU(512)
#         self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)
#         self.fc_mu = nn.Linear(512, latent_dim)
#         self.fc_logvar = nn.Linear(512, latent_dim)

#     def forward(self, x):
#         x = self.relu1_1(self.conv1_1(x))
#         x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

#         x = self.relu2_1(self.conv2_1(x))
#         x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
#         x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

#         x = self.relu3_1(self.conv3_1(x))
#         x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
#         x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
#         x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
#         x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

#         x = self.relu4_1(self.conv4_1(x))
#         x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))
#         x = self.pooling(x)
#         x = x.view(x.size(0), -1)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)

#         return mu, logvar

# 生成一个 [241*96] 的张量
class Generator(nn.Module):
    def __init__(self, id_dim=256, a_dim=256, out_size=(241, 96)):
        super(Generator, self).__init__()

        self.out_size = out_size

        self.fc = nn.Linear(id_dim + a_dim, 512)
        self.up = nn.Upsample(scale_factor=8)

        self.block1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block2 = ResBlock(256, 256)
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block4 = ResBlock(256, 256)
        self.block5 = ResBlock(256, 256)
        self.block6 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.LeakyReLU()
        )
        self.block7 = ResBlock(128, 128)
        self.block8 = ResBlock(128, 64)
        self.block9 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
            # nn.LeakyReLU()
            # nn.Tanh()
        )

    def forward(self, identity, attribute):
        x = torch.cat((identity, attribute), 1)
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
        x = x[:, :, :self.out_size[0], :self.out_size[1]] # 实际只使用 [n, 1, 241, 96]
        x = x.view(x.size(0), x.size(2), x.size(3)) # [n, 241, 96]
        return x

class Discriminator(nn.Module):
    def __init__(self, input_size, level_channel_num=32, level_num=6, kernel_size=2, dropout=0.2):
        super(Discriminator, self).__init__()

        channels = [level_channel_num] * level_num
        self.tcn = TemporalConvNet(
            input_size, # 骨骼1帧 + 动作240帧
            channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        self.fc1 = nn.Linear(channels[-1], 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, data):
        # [n, 241, 96]
        x = self.tcn(data)
        x = x[:, :, -1] # [n, 256]
        mid = F.leaky_relu(self.fc1(x), negative_slope=0.2) # [n, 64]
        p = self.fc2(mid) # [n, 1]
        p = torch.sigmoid(p) # [n, 1]
        p = p.view(p.size(0)) # [n]

        # x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        # x = F.leaky_relu(self.BN2(self.conv2(x)), negative_slope=0.2)
        # x = F.leaky_relu(self.BN3(self.conv3(x)), negative_slope=0.2)
        # x = x.view(-1, self.featmap_dim * 4 * 4)
        # out = F.sigmoid(self.fc(x))
        return p, mid # 返回此真概率、中间特征向量
