import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import Encoder
from model import Classifier
from model import Generator
from model import Discriminator
import dataloader
import data_utils
import pathlib
from datetime import datetime

output_path = os.path.join('.', 'outputs', datetime.now().strftime('%Y-%m-%d'))
pathlib.Path(os.path.join(output_path, 'models')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(output_path, 'gens')).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = dataloader.MyDataset(*dataloader.load_dir_data_statistics_category_num(dataset_dir='./v5/walk_id_compacted'))

batch_size = 20
learning_rate = 1e-4
epochs_num = 1000000
category_num = dataset.category_num
z_dim = 32

# 重构参数
lbd_1 = 3
lbd_2 = 1
lbd_3 = 1e-3
lbd_4 = 1e-3

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

E = Encoder(
  in_channel_num=96, # 96 个关节通道
  z_dim=z_dim
).to(device)

G = Generator(
  z_dim=z_dim,
  c_dim=category_num,
  out_size=(96, 240)
).to(device)

D = Discriminator(
  in_channel_num=96, # 96 个关节通道
  f_d_dim=32
).to(device)

C = Classifier(
  in_channel_num=96, # 96 个关节通道
  f_c_dim=32,
  category_num=category_num
).to(device)

def save_models(dirpath):
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
  torch.save(E.state_dict(), os.path.join(dirpath, 'E.pt'))
  torch.save(G.state_dict(), os.path.join(dirpath, 'G.pt'))
  torch.save(D.state_dict(), os.path.join(dirpath, 'D.pt'))
  torch.save(C.state_dict(), os.path.join(dirpath, 'C.pt'))

def load_models(dirpath):
  E.load_state_dict(torch.load(os.path.join(dirpath, 'E.pt')))
  G.load_state_dict(torch.load(os.path.join(dirpath, 'G.pt')))
  D.load_state_dict(torch.load(os.path.join(dirpath, 'D.pt')))
  C.load_state_dict(torch.load(os.path.join(dirpath, 'C.pt')))

# load_models(os.path.join('outputs', '2021-01-28', 'models', '轮{}'.format(5249)))

optimizer_E = torch.optim.Adam(E.parameters(), learning_rate)
optimizer_G = torch.optim.Adam(G.parameters(), learning_rate)
optimizer_D = torch.optim.Adam(D.parameters(), learning_rate)
optimizer_C = torch.optim.Adam(C.parameters(), learning_rate)

cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()
def kl_loss(mean, logvar):
  return -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

legends = ['L_C', 'L_KL', 'L_D', 'L_GD', 'L_GC', 'L_G']
losses = np.array([]).reshape(0, len(legends)) # 用于绘制折线图

# 训练
total_batch = 0
for epoch in range(epochs_num):
  for batch_i, data in enumerate(dataloader):
    total_batch += 1
    print('## 轮次：{} 批次：{} 总批：{}'.format(epoch, batch_i, total_batch))

    skeleton, frames, label = data

    # 从真实样本中取样 {x_r, c_r} ~ P_r
    x_r = frames.to(device) # [N, 96, 240]

    c_r = torch.zeros((x_r.shape[0], category_num)) # 转为 onehot
    c_r[torch.arange(x_r.shape[0]), label] = 1
    c_r = c_r.to(device)

    # 训练分类器 C
    f_c_x_r, c_x_r = C(x_r)

    L_C = bce_loss(c_x_r, c_r)
    C.zero_grad()
    L_C.backward()
    optimizer_C.step()
    print('L_C', L_C.item())

    f_c_x_r.detach_()

    # 训练 D
    f_d_x_r, d_x_r = D(x_r)

    z, mean, logvar = E(x_r) #, c_r)
    L_KL = kl_loss(mean, logvar)
    x_f = G(z, c_r)
    f_d_x_f, d_x_f = D(x_f)
    print('L_KL', L_KL.item())

    # 取样随机噪声 z_p ~ P_z，取样随机分类 c_p
    z_p = torch.randn(x_r.shape[0], z_dim).to(device)
    c_p_n = torch.randint(0, category_num, size=(x_r.shape[0],))
    c_p = torch.nn.functional.one_hot(c_p_n, category_num).float().to(device)
    x_p = G(z_p, c_p)
    f_d_x_p, d_x_p = D(x_p)
    L_D = bce_loss(d_x_r, torch.ones_like(d_x_r)) + bce_loss(d_x_f, torch.zeros_like(d_x_f)) + bce_loss(d_x_p, torch.zeros_like(d_x_p))
    
    D.zero_grad()
    L_D.backward(retain_graph=True)
    optimizer_D.step()
    print('L_D', L_D.item())

    # 训练 G、E
    # 计算 x_r 和 x_p 的特征中心 f_d
    L_GD = mse_loss(torch.mean(f_d_x_r, dim=0), torch.mean(f_d_x_p, dim=0))
    print('L_GD', L_GD.item())

    # 计算 x_r 和 x_p 的特征中心 f_c
    f_c_x_p, _ = C(x_p)
    L_GC = mse_loss(torch.mean(f_c_x_r, dim=0), torch.mean(f_c_x_p, dim=0))
    print('L_GC', L_GC.item())

    f_c_x_f, _ = C(x_f)
    L_G = mse_loss(x_r, x_f) + mse_loss(f_d_x_r, f_d_x_f) + mse_loss(f_c_x_r, f_c_x_f)
    print('L_G', L_G.item())

    L_Gs = lbd_2 * L_G + lbd_3 * L_GD + lbd_4 * L_GC
    G.zero_grad()
    L_Gs.backward(retain_graph=True)
    optimizer_G.step()
    print('L_Gs', L_Gs.item())

    L_Es = lbd_1 * L_KL + lbd_2 * L_G
    E.zero_grad()
    L_Es.backward()
    optimizer_E.step()
    print('L_Es', L_Es.item())

    ############################### 画损失图
    losses = np.concatenate((losses, np.array([L_C.item(), L_KL.item(), L_D.item(), L_GD.item(), L_GC.item(), L_G.item()]).reshape(1, -1)), axis=0)
    losses = losses[-10000:, :] # 只查看最近的损失
    x_axis = np.arange(losses.shape[0])

    fig, axs = plt.subplots(losses.shape[1])
    for i in range(len(axs)):
      axs[i].set_title(legends[i])
      axs[i].plot(x_axis, losses[:, i])
    fig.savefig('losses.png')
    plt.close(fig)

    # 输出检查点
    if total_batch % 500 == 0:
      # statistics = np.loadtext('./v5/walk_id_compacted/_min_max_mean_std.csv')

      frames = np.array([data_utils.normalized_frames_to_frames(x_p.detach().cpu().numpy(), dataset.statistics) for x in frames])

      # np.savetxt('./test.csv', x_f[0].detach().cpu().numpy())

      data_utils.save_bvh_to_file(
        os.path.join(output_path, 'gens', '轮{}-批{}-标{}-P.bvh'.format(epoch, batch_i, c_p_n[0])),
        skeleton[0], # 先随便贴个骨骼吧
        frames[0]
      )

      # 保存模型
      save_models(
        os.path.join(output_path, 'models', '轮{}'.format(epoch))
      )
