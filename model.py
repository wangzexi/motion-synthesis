import torch
import torch.nn as nn

# 分类器 C
class Classifier(nn.Module):
    def __init__(self, in_channel_num, f_c_dim=24, category_num=10):
        super(Classifier, self).__init__()

        self.gru = nn.GRU(
            input_size=in_channel_num,
            hidden_size=f_c_dim,
            num_layers=4
        )

        self.fc0 = nn.Sequential(
            nn.Linear(f_c_dim, f_c_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(f_c_dim, category_num),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, C, T]
        x = data.permute([2, 0, 1]) # [T, N, C]
        _, h = self.gru(x)
        f_c = self.fc0(h[-1, :, :])

        x = self.fc1(f_c) # one-hot [N, category_num]
        return f_c, x

# 编码器
class Encoder(nn.Module):
    def __init__(self, in_channel_num, z_dim):
        super(Encoder, self).__init__()

        self.gru = nn.GRU(
            input_size=in_channel_num,
            hidden_size=64,
            num_layers=4
        )
        self.fc0 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2, True)
        )
        self.fc1 = nn.Linear(64, z_dim)
        self.fc2 = nn.Linear(64, z_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(mean)
        return mean + eps * std

    def forward(self, data):
        # data: [N, C, T]
        x = data.permute([2, 0, 1]) # [T, N, C]
        _, h = self.gru(x)
        x = self.fc0(h[-1, :, :])

        mean = self.fc1(x)
        logvar = self.fc2(x)
        z = self.reparameterize(mean, logvar)
        return z, mean, logvar

# 生成 [239, 96]
class Generator(nn.Module):
    def __init__(self, z_dim=256, c_dim=256, out_channel_num=96, seq_len=239):
        super(Generator, self).__init__()
        self.input_size = out_channel_num
        self.hidden_size = out_channel_num
        self.seq_len = seq_len
        self.num_layers = 4

        self.fc1 = nn.Sequential(
            nn.Linear(z_dim + c_dim, self.hidden_size * self.num_layers),
            nn.LeakyReLU(0.2, True)
        )
        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh()
        )

    def forward(self, z, c):
        x = torch.cat((z, c), dim=1)
        h = self.fc1(x)
        h = h.view(self.num_layers, x.size(0), -1)

        out, h = self.gru(
            torch.zeros((self.seq_len, x.size(0), self.input_size), device=x.device),
            h
        )
        x = self.fc2(out)
        return out.permute([1, 2, 0]) # [N, C, T]

class Discriminator(nn.Module):
    def __init__(self, in_channel_num, f_d_dim=64):
        super(Discriminator, self).__init__()

        self.gru = nn.GRU(
            input_size=in_channel_num,
            hidden_size=f_d_dim,
            num_layers=4
        )

        self.fc0 = nn.Sequential(
            nn.Linear(f_d_dim, f_d_dim),
            nn.LeakyReLU(0.2, True)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(f_d_dim, f_d_dim),
            nn.Sigmoid()
        )

    def forward(self, data):
        # data: [N, 96, 239]
        x = data.permute([2, 0, 1]) # [T, N, C]
        _, h = self.gru(x)
        f_d = self.fc0(h[-1, :, :])

        x = self.fc1(f_d)
        x = x.squeeze(1) # [N]
        return f_d, x
