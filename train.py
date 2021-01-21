import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EncoderTCN
from model import AttributeTCN
from model import GeneratorTCN
from model import Generator
from model import Discriminator
import dataloader
import data_utils
import pathlib

from datetime import datetime

output_path = os.path.join('.', 'outputs', datetime.now().strftime('%Y-%m-%d'))
print(output_path)

pathlib.Path(os.path.join(output_path, 'models')).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(output_path, 'gens')).mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = dataloader.Hybrid_Dataset(*dataloader.load_dir_data_statistics_category_num(dataset_dir='./v5/walk_id_compacted'))

batch_size = 20
learning_rate = 1e-4
epochs_num = 100
category_num = dataset.category_num

CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter

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
  in_channel_num=96, # 96 个关节通道
  identity_dim=64,
  category_num=category_num,
  kernel_size=5,
  dropout=0.3
).to(device)

I_path = os.path.join(output_path, 'I.pt')
if os.path.isfile(I_path): # 如果有预训练的 I，就直接载入
  I.load_state_dict(torch.load(I_path))
  print('已经载入预训练的 I.pt 文件')
else:
  print('不存在预训练的 I.pt 文件')
  exit()

A = AttributeTCN(
  in_channel_num=96,
  out_dim=64,
  kernel_size=5,
  dropout=0.2
).to(device)

G = Generator(
  identity_dim=64,
  attribute_dim=64,
  out_size=(96, 239)
  # kernel_size=5,
  # dropout=0.2
).to(device)

D = Discriminator(
  in_channel_num=96, # 96 个关节通道
  level_channel_num=256, # 每层特征提取器数量 256
  level_num=8,
  kernel_size=5,
  dropout=0.2
).to(device)

# load_models('./models/轮次40')

# VAE 的采样
def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

# WGAN-GP 的梯度惩罚
def calc_gradient_penalty(netD, real_data, fake_data):
  alpha = torch.rand(real_data.size(0), 1).to(device)
  alpha = alpha.expand((real_data.size(0), real_data.size(1) * real_data.size(2)))
  alpha = alpha.view(real_data.size(0), real_data.size(1), real_data.size(2))

  interpolates = alpha * real_data + ((1 - alpha) * fake_data)
  interpolates.requires_grad_(True)

  disc_interpolates, _ = netD(interpolates)

  gradients = torch.autograd.grad(
    outputs=disc_interpolates,
    inputs=interpolates,
    grad_outputs=torch.ones_like(disc_interpolates),
    create_graph=True,
    retain_graph=True,
    only_inputs=True
  )[0]

  gradients = gradients.view(gradients.size(0), -1)
  gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
  return gradient_penalty

a_optimizer = torch.optim.Adam(A.parameters(), learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), learning_rate)

cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()

legends = ['loss_I freeze', 'loss_KL', 'loss_D', 'loss_GD', 'loss_GR', 'loss_GC']
losses = np.array([]).reshape(0, len(legends)) # 用于绘制折线图

# 训练
for epoch in range(epochs_num):
  for batch_idx, (data_a, data_b) in enumerate(dataloader):
    skeleton_s, frames_s, label_s = data_a
    skeleton_a, frames_a, label_a = data_b

    # 排除掉第一帧，因为第一帧不是增量
    x_s = frames_s[:, :, 1:].to(device=device) # [N, 96, 239]
    x_a = frames_a[:, :, 1:].to(device=device)
    
    label_s = label_s.to(device=device)

    print('# 轮次：{}，批次：{}'.format(epoch, batch_idx))
    ## 自交杂交轮流进行
    if (batch_idx % 2 == 1):
      # 奇数步自交
      lbd = 1 # 参数 λ，控制自交杂交重构相似程度
      x_a = x_s
      skeleton_a = skeleton_s
      print('## 自交，λ：{}'.format(lbd))
    else:
      # 偶数步杂交
      lbd = 0.1
      print('## 杂交，λ：{}'.format(lbd))

    ################## 训练 D

    i_ctg, i_id = I(x_s)
    loss_I = cross_entropy_loss(i_ctg, label_s)
    print('loss_I', loss_I.item())

    for _ in range(CRITIC_ITERS): # 更多的训练 D
      a_mu, a_log_var = A(x_a)
      a_z = reparameterize(a_mu, a_log_var) # z ~ N

      x_f = G(i_id, a_z) # f 代表 fake

      d_real_critic, _ = D(x_a)
      d_fake_critic, _ = D(x_f)
      loss_D_critic = -(torch.mean(d_real_critic) - torch.mean(d_fake_critic))
      print('loss_D_critic', loss_D_critic.item())

      gradient_penalty = calc_gradient_penalty(D, x_a, x_f) # 这里用 a 到 f 插值

      loss_1 = loss_D_critic + gradient_penalty
      d_optimizer.zero_grad()
      loss_1.backward(retain_graph=True)
      d_optimizer.step()

    ################## 训练 A, G
    _, i_id = I(x_s)
    a_mu, a_log_var = A(x_a)
    a_z = reparameterize(a_mu, a_log_var)
    loss_KL = torch.mean(0.5 * (torch.pow(a_mu, 2) + torch.exp(a_log_var) - a_log_var - 1))
    print('loss_KL', loss_KL.item())

    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_a)
    d_fake_p, d_fake_feature = D(x_f)
    loss_GD = 0.5 * mse_loss(d_fake_feature, d_real_feature)
    loss_GR = lbd * 0.5 * mse_loss(x_f, x_a)
    print('loss_GD', loss_GD.item())
    print('loss_GR', loss_GR.item())

    c_ctg, c_fake_id = I(x_f) # 这里原本是用 C 再识别，现在改为 I
    loss_GC = 0.5 * mse_loss(c_fake_id, i_id)
    loss_C = cross_entropy_loss(c_ctg, label_s)
    print('loss_GC', loss_GC.item())
    print('loss_C', loss_C.item())

    a_optimizer.zero_grad()
    g_optimizer.zero_grad()
    loss_2 = loss_KL + loss_GR + loss_GD + loss_GC + loss_C
    loss_2.backward()
    a_optimizer.step()
    g_optimizer.step()

    ############################### 画损失图
    losses = np.concatenate((losses, np.array([loss_I.item(), loss_KL.item(), loss_D_critic.item(), loss_GD.item(), loss_GR.item(), loss_GC.item()]).reshape(1, -1)), axis=0)
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
        skeleton = skeleton_s.numpy() # [N, 93]
        base_frames = frames_s[:, :, 0:1]
      else:
        tag = '杂交'
        label = label_a
        skeleton = skeleton_a.numpy() # [N, 93]
        base_frames = frames_a[:, :, 0:1]

      frames = torch.cat((base_frames, x_f.detach().cpu()), dim=2).numpy() # 拼上原始第一帧
      frames = np.array([data_utils.normalized_frames_to_frames(x, dataset.statistics) for x in frames])
      frames = np.array([data_utils.transform_detal_frames_to_frames(x) for x in frames]) # [N, 96, 240]

      # np.savetxt('./test.csv', x_f[0].detach().cpu().numpy())

      data_utils.save_bvh_to_file(
        os.path.join(output_path, 'gens', '轮次{}-批次{}-{}-ID{}.bvh'.format(epoch, batch_idx, tag, label[0])),
        skeleton[0],
        frames[0]
      )

    # 保存模型
    if epoch % 1 == 0 and batch_idx == 0:
      save_models(
        os.path.join(output_path, 'models', '轮次{}'.format(epoch))
      )
