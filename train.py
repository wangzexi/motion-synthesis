import os
import torch
from model import EncoderTCN
from model import AttributeTCN
from model import Generator
from model import Discriminator
from dataloader import MyDataset
import data_utils
import pathlib

def save_models(dirpath):
  pathlib.Path(dirpath).mkdir(parents=True, exist_ok=True)
  torch.save(I.state_dict(), os.path.join(dirpath, 'I.pt'))
  torch.save(C.state_dict(), os.path.join(dirpath, 'C.pt'))
  torch.save(A.state_dict(), os.path.join(dirpath, 'A.pt'))
  torch.save(D.state_dict(), os.path.join(dirpath, 'D.pt'))
  torch.save(G.state_dict(), os.path.join(dirpath, 'G.pt'))

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=30, shuffle=True, drop_last=True)

learning_rate = 1e-4
num_epochs = 100
num_classes = dataset.category_num

# 先预训练 identity 分类器
# I = EncoderTCN(
#   input_size=241, # 骨骼1帧 + 动作240帧
#   latent_dim=64,
#   category_num=num_classes,
#   level_channel_num=256,
#   level_num=7,
#   kernel_size=3,
#   dropout=0.2
# ).to(device)

# i_path = './models/I.pt'
# if os.path.isfile(i_path): # 如果有预训练的 I，就直接载入，跳过训练
#   I.load_state_dict(torch.load(i_path))
# else:
#   li_loss = torch.nn.CrossEntropyLoss()
#   i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)

#   # 训练 I
#   for epoch in range(num_epochs):
#     for batch_idx, (data, targets) in enumerate(dataloader):
#       data = data.to(device=device)
#       targets = targets.to(device=device)
      
#       class_pred, identity = I(data)
#       i_loss = li_loss(class_pred, targets)

#       i_optimizer.zero_grad()
#       i_loss.backward()
#       i_optimizer.step()

#       print('epoch {} batch_idx {} LOSS {}'.format(epoch, batch_idx, i_loss))
  
#   torch.save(I.state_dict(), i_path)

# if 1 == 0: # 手动测试开关
#   # 分类器测试，这是个错误的测试方法，它用作测试的是训练集
#   for data, targets in dataloader:
#     data = data.to(device=device)
#     targets = targets.to(device=device)
#     class_pred, identity = I(data)
    
#     p = torch.nn.functional.softmax(class_pred)
#     print('----')
#     print(p.max(dim=1)) #, targets)
#     print(targets)

def reparameterize(mu, logvar):
  std = torch.exp(0.5 * logvar)
  eps = torch.randn_like(std)
  return eps * std + mu

I = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  category_num=num_classes,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)

A = AttributeTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)

G = Generator(
  id_dim=64,
  a_dim=64
).to(device)

C = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  latent_dim=64,
  category_num=num_classes,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.2
).to(device)
# C.load_state_dict(torch.load(i_path)) # 直接使用预训练好的 I

D = Discriminator(
  input_size=241, # 骨骼1帧 + 动作240帧
  level_channel_num=256, # 提取器 [256]
  level_num=8,
  kernel_size=3,
  dropout=0.2
).to(device)

i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)
a_optimizer = torch.optim.Adam(A.parameters(), learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), learning_rate)
c_optimizer = torch.optim.Adam(C.parameters(), learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), learning_rate)

cross_entropy_loss = torch.nn.CrossEntropyLoss()
bce_loss = torch.nn.BCELoss()
mse_loss = torch.nn.MSELoss()

# 联合训练
for epoch in range(num_epochs):
  for batch_idx, (data_a, labels_a, data_b, labels_b) in enumerate(dataloader):

    batch = data_a.size(0)

    x_s = data_a.to(device=device)
    x_a = data_b.to(device=device)
    labels_a = labels_a.to(device=device)
    # labels_b = labels_b.to(device=device) # 暂时用不到

    print('# 轮次：{}，批次：{}'.format(epoch, batch_idx))
    ## 自交杂交轮流进行
    if (batch_idx % 2 == 1):
      # 奇数步自交
      lbd = 1 # 参数 λ
      x_a = x_s
      print('## 自交，λ：{}'.format(lbd))
    else:
      # 偶数步杂交
      lbd = 0.1
      print('## 杂交，λ：{}'.format(lbd))

    ################## 训练 I、C、A、D

    i_ctg, i_id = I(x_s) # ctg 代表 category
    c_ctg, _ = C(x_s)
    loss_I = cross_entropy_loss(i_ctg, labels_a)
    loss_C = cross_entropy_loss(c_ctg, labels_a)
    print('loss_I', loss_I.item())
    print('loss_C', loss_C.item())

    a_mu, a_log_var = A(x_a) # VAE 的均值和标准差
    a_z = reparameterize(a_mu, a_log_var) # 从均值和标准差组成的正态分布里采出一个 z
    loss_KL = torch.mean(0.5 * (torch.pow(a_mu, 2) + torch.exp(a_log_var) - a_log_var - 1)) # 贴近正态分布的约束
    print('loss_KL', loss_KL.item())

    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_s)
    d_fake_p, d_fake_feature = D(x_f)
    # loss_D = torch.mean(-torch.log(d_real_p) - torch.log(1 - d_fake_p)) # 会出 log0 nan
    loss_D = (bce_loss(d_real_p, torch.ones_like(d_real_p)) + bce_loss(d_fake_p, torch.zeros_like(d_fake_p))) / 2

    print('loss_D', loss_D.item())

    i_optimizer.zero_grad()
    c_optimizer.zero_grad()
    d_optimizer.zero_grad()
    a_optimizer.zero_grad()

    loss_1 = loss_I + loss_C + loss_KL + loss_D
    loss_1.backward()
    i_optimizer.step()
    c_optimizer.step()
    d_optimizer.step()
    a_optimizer.step()

    ################## 训练 G
    _, i_id = I(x_s) # ctg 代表 category
    a_mu, a_log_var = A(x_a) # VAE 的均值和标准差
    a_z = reparameterize(a_mu, a_log_var) # 贴上一个正态分布
    x_f = G(i_id, a_z) # f 代表 fake

    d_real_p, d_real_feature = D(x_s)
    d_fake_p, d_fake_feature = D(x_f)
    # loss_GD = torch.mean(torch.diagonal(torch.cdist(d_fake_feature, d_real_feature)) * (1 / 2)) # 这里计算了n²个距离，然后抓出对角线，有性能优化空间
    # loss_GR = torch.mean(torch.diagonal(torch.cdist(x_f.reshape(batch, -1), x_a.view(batch, -1))) * (lbd / 2)) # 把 x_a 展平，计算 x_a 与 x_f 平方差距离
    loss_GD = (1/2) * mse_loss(d_fake_feature, d_real_feature)
    loss_GR = (lbd/2) * mse_loss(x_f.reshape(batch, -1), x_a.view(batch, -1))

    print('loss_GD', loss_GD.item())
    print('loss_GR', loss_GR.item())

    _, c_fake_id = C(x_f)
    # loss_GC = torch.mean(torch.diagonal(torch.cdist(c_fake_id, i_id)) * (1 / 2))
    loss_GC = (1/2) * mse_loss(c_fake_id, i_id)
    print('loss_GC', loss_GC.item())

    g_optimizer.zero_grad()
    loss_2 = lbd * loss_GR + loss_GD + loss_GC
    loss_2.backward()
    g_optimizer.step()

    if epoch % 5 == 0 and batch_idx in [1, 2]:
      data_utils.save_bvh_to_file(
        './outputs/轮次{}-批次{}-{}.bvh'.format(epoch, batch_idx, '自交' if batch_idx % 2 == 1 else '杂交'),
        x_f[0].cpu().detach()
      )
    if epoch % 10 == 0 and batch_idx == 0:
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
