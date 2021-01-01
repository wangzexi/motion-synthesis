import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EncoderTCN
from model import AttributeTCN
from model import Generator
from model import Discriminator
import dataloader
import data_utils
import pathlib

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

dataset = dataloader.Hybrid_Dataset(*dataloader.load_dir_data_statistics_category_num(dataset_dir='./v5/walk_id_compacted'))

batch_size = 20
learning_rate = 1e-4
num_epochs = 100
num_classes = dataset.category_num

dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True)

def save_models(dirpath):
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
  torch.save(A.state_dict(), os.path.join(dirpath, 'A.pt'))
  torch.save(D.state_dict(), os.path.join(dirpath, 'D.pt'))
  torch.save(G.state_dict(), os.path.join(dirpath, 'G.pt'))

def load_models(dirpath):
  A.load_state_dict(torch.load(os.path.join(dirpath, 'A.pt')))
  D.load_state_dict(torch.load(os.path.join(dirpath, 'D.pt')))
  G.load_state_dict(torch.load(os.path.join(dirpath, 'G.pt')))

I = EncoderTCN(
  input_size=239, # frames 239
  latent_dim=64,
  category_num=num_classes,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.3
).to(device)

I_path = './models/I.pt'
if os.path.isfile(I_path): # 如果有预训练的 I，就直接载入
  I.load_state_dict(torch.load(I_path))
  print('已经载入预训练的 I.pt 文件')
else:
  print('不存在预训练的 I.pt 文件')
  exit()

A = AttributeTCN(
  input_size=239, # frames 239
  latent_dim=64,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)

G = Generator(
  id_dim=64,
  a_dim=64,
  out_size=(239, 96)
).to(device)

# C = EncoderTCN(
#   input_size=241, # 骨骼1帧 + 动作240帧
#   latent_dim=64,
#   category_num=num_classes,
#   level_channel_num=256,
#   level_num=7,
#   kernel_size=3,
#   dropout=0.2
# ).to(device)

D = Discriminator(
  input_size=239, # 骨骼1帧 + 动作240帧
  level_channel_num=256, # 提取器 [256]
  level_num=8,
  kernel_size=3,
  dropout=0.2
).to(device)

# load_models('./models/轮次40')

def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)
a_optimizer = torch.optim.Adam(A.parameters(), learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), learning_rate)

cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()

legends = ['loss_I freeze', 'loss_KL', 'loss_D', 'loss_GD', 'loss_GR', 'loss_GC']
losses = np.array([]).reshape(0, len(legends)) # 用于绘制折线图

# 训练
for epoch in range(num_epochs):
  for batch_idx, (data_a, data_b) in enumerate(dataloader):
    skeleton_s, frames_s, label_s = data_a
    skeleton_a, frames_a, label_a = data_b

    # 排除掉第一帧，因为第一帧不是增量
    x_s = frames_s[:, 1:, :].to(device=device) # [N, 239, 96]
    x_a = frames_a[:, 1:, :].to(device=device)
    
    label_s = label_s.to(device=device)

    print('# 轮次：{}，批次：{}'.format(epoch, batch_idx))
    ## 自交杂交轮流进行
    if (batch_idx % 2 == 1):
      # 奇数步自交
      lbd = 1 # 参数 λ
      x_a = x_s
      skeleton_a = skeleton_s
      print('## 自交，λ：{}'.format(lbd))
    else:
      # 偶数步杂交
      lbd = 0.1
      print('## 杂交，λ：{}'.format(lbd))

    ################## 训练 I、A、D

    i_ctg, i_id = I(x_s)
    loss_I = cross_entropy_loss(i_ctg, label_s)
    print('loss_I', loss_I.item())

    a_mu, a_log_var = A(x_a) # VAE 的均值和标准差
    a_z = reparameterize(a_mu, a_log_var) # 从均值和标准差组成的正态分布里采出一个 z
    loss_KL = torch.mean(0.5 * (torch.pow(a_mu, 2) + torch.exp(a_log_var) - a_log_var - 1)) # 贴近正态分布的约束
    print('loss_KL', loss_KL.item())

    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_a)
    d_fake_p, d_fake_feature = D(x_f)
    # loss_D = torch.mean(-torch.log(d_real_p) - torch.log(1 - d_fake_p)) # 会出 log0 nan
    loss_D = 0.5 * (bce_loss(d_real_p, torch.ones_like(d_real_p)) + bce_loss(d_fake_p, torch.zeros_like(d_fake_p)))

    print('loss_D', loss_D.item())

    # i_optimizer.zero_grad() # 不再更新 I
    d_optimizer.zero_grad()
    a_optimizer.zero_grad()

    loss_1 = loss_I + loss_KL + loss_D
    loss_1.backward()
    # i_optimizer.step()
    d_optimizer.step()
    a_optimizer.step()

    ################## 训练 G
    _, i_id = I(x_s)
    a_mu, a_log_var = A(x_a) # VAE 的均值和标准差
    a_z = reparameterize(a_mu, a_log_var) # 贴上一个正态分布
    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_a)
    d_fake_p, d_fake_feature = D(x_f)
    loss_GD = 0.5 * mse_loss(d_fake_feature, d_real_feature)
    # loss_GR = lbd * 0.5 * mse_loss(x_f.reshape(batch_size, -1), x_a.view(batch_size, -1))
    loss_GR = lbd * 0.5 * mse_loss(x_f, x_a)

    print('loss_GD', loss_GD.item())
    print('loss_GR', loss_GR.item())

    c_ctg, c_fake_id = I(x_f) # 这里原本是用 C 再识别，现在改为 I
    loss_C = cross_entropy_loss(c_ctg, label_s)
    print('loss_C', loss_C.item())
    loss_GC = 0.5 * mse_loss(c_fake_id, i_id)
    print('loss_GC', loss_GC.item())

    g_optimizer.zero_grad()
    loss_2 = loss_GR + loss_GD + loss_GC + loss_C
    loss_2.backward()
    g_optimizer.step()

    # 画损失图
    losses = np.concatenate((losses, np.array([loss_I.item(), loss_KL.item(), loss_D.item(), loss_GD.item(), loss_GR.item(), loss_GC.item()]).reshape(1, -1)), axis=0)
    losses = losses[-10000:, :] # 只查看最近的损失
    x_axis = np.arange(losses.shape[0])

    fig, axs = plt.subplots(losses.shape[1])
    for i in range(len(axs)):
      axs[i].set_title(legends[i])
      axs[i].plot(x_axis, losses[:, i])
    fig.savefig('losses.png')
    plt.close(fig)

    # 输出检查点
    if epoch % 1 == 0 and (batch_idx % 1000 == 0 or batch_idx % 1000 == 1):
      # statistics = np.loadtext('./v5/walk_id_compacted/_min_max_mean_std.csv')

      if batch_idx % 2 == 1:
        tag = '自交'
        label = label_s
        skeleton = skeleton_s.view(batch_size, 1, -1).numpy() # [N, 1, 96]
        base_frames = frames_s[:, 0:1, :]
      else:
        tag = '杂交'
        label = label_a
        skeleton = skeleton_a.view(batch_size, 1, -1).numpy() # [N, 1, 96]
        base_frames = frames_a[:, 0:1, :]

      frames = torch.cat((base_frames, x_f.detach().cpu()), dim=1).numpy() # 拼上原始第一帧
      frames = np.array([data_utils.normalized_frames_to_frames(x, dataset.statistics) for x in frames])
      frames = np.array([data_utils.transform_detal_frames_to_frames(x) for x in frames])
      data = np.concatenate((skeleton, frames), axis=1) # [N, 241, 96]

      # np.savetxt('./test.csv', x_f[0].detach().cpu().numpy())

      data_utils.save_bvh_to_file(
        './outputs/轮次{}-批次{}-{}-ID{}.bvh'.format(epoch, batch_idx, tag, label[0]),
        data[0]
      )

    # 保存模型
    if epoch % 1 == 0 and batch_idx == 0:
      save_models('./models/轮次{}'.format(epoch))

    ## 上面是原始代码的训练步骤
    ## 下面是论文中的训练步骤
    ## 都会遇到原地操作 Good Luck 报错
    ## 计算慢可以开启在第 11 行开启 gpu

    # i_optimizer.zero_grad()
    # loss_I.backward(retain_graph=True)
    # i_optimizer.step()

    # c_optimizer.zero_grad()
    # loss_C.backward(retain_graph=True)
    # c_optimizer.step()

    # d_optimizer.zero_grad()
    # loss_D.backward(retain_graph=True)
    # d_optimizer.step()

    # g_optimizer.zero_grad()
    # loss_G = lbd * loss_GR + loss_GD + loss_GC
    # loss_G.backward(retain_graph=True)
    # g_optimizer.step()

    # loss_A = lbd * (loss_KL + loss_G)
    # loss_A.backward()
    # a_optimizer.step()
