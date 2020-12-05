import os
import torch
from model import EncoderTCN
from model import AttributeTCN
from model import Generator
from model import Discriminator
from dataloader import MyDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MyDataset()
dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=10, shuffle=True, drop_last=True)

learning_rate = 1e-3
num_epochs = 10
num_classes = dataset.categoryCount

# 先预训练 identity 分类器
I = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  output_size=num_classes,
  num_channels=256, # identity [256]
  levels=7,
  kernel_size=3,
  dropout=0.2
).to(device)

i_path = './models/I.pt'
if os.path.isfile(i_path): # 如果有预训练的 I，就直接载入，跳过训练
  I.load_state_dict(torch.load(i_path))
else:
  li_loss = torch.nn.CrossEntropyLoss()
  i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)

  # 训练 I
  for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(dataloader):
      data = data.to(device=device)
      targets = targets.to(device=device)
      
      class_pred, identity = I(data)
      i_loss = li_loss(class_pred, targets)

      i_optimizer.zero_grad()
      i_loss.backward()
      i_optimizer.step()

      print('epoch {} batch_idx {} LOSS {}'.format(epoch, batch_idx, i_loss))
  
  torch.save(I.state_dict(), i_path)

if 1 == 0: # 手动测试开关
  # 分类器测试，这是个错误的测试方法，它用作测试的是训练集
  for data, targets in dataloader:
    data = data.to(device=device)
    targets = targets.to(device=device)
    class_pred, identity = I(data)
    
    p = torch.nn.functional.softmax(class_pred)
    print('----')
    print(p.max(dim=1)) #, targets)
    print(targets)


# 正经网络训练

A = AttributeTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  num_channels=256, # Attribute [256]
  levels=10,
  kernel_size=3,
  dropout=0.2
).to(device)

G = Generator(
  id_dim=256,
  a_dim=256
).to(device)

C = EncoderTCN(
  input_size=241, # 骨骼1帧 + 动作240帧
  output_size=num_classes,
  num_channels=256, # identity [256]
  levels=7,
  kernel_size=3,
  dropout=0.2
).to(device)
C.load_state_dict(torch.load(i_path)) # 直接使用预训练好的 I

D = Discriminator(
  input_size=241, # 骨骼1帧 + 动作240帧
  num_channels=256, # 提取器 [256]
  levels=8,
  kernel_size=3,
  dropout=0.2
).to(device)

# kl_loss = torch.nn.CrossEntropyLoss() # 需要改成 KL 散度约束 A 的向量分布尽可能与 I 不同
# gr_loss = torch.nn.MSELoss() # MSE loss
# c_loss = -log(C(x^')向量的第x^s的标签个元素) / ||fc(x^')-fc(x^s)||^2
# d_loss = -log(D(x^a)) -log(1-D(G(身份向量, 特征向量)))

a_optimizer = torch.optim.Adam(A.parameters(), learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), learning_rate)
c_optimizer = torch.optim.Adam(C.parameters(), learning_rate)
d_optimizer = torch.optim.Adam(D.parameters(), learning_rate)

# 联合训练
for epoch in range(num_epochs):
  for batch_idx, (data, targets) in enumerate(dataloader):
    batch = data.size(0)
    data = data.to(device=device)
    targets = targets.to(device=device)

    _, i_identity = I(data) 
    attribute = A(data) # 让身份序列和特征序列相同，可能方便训练A

    real = data
    fake = G(i_identity, attribute)

    # L_GR
    # g_loss = 0.5*||fc(x^a)-fc(x^')||^2
    real_flatten = real.view(batch, -1)
    fake_flatten = fake.reshape(batch, -1)
    g_loss = torch.diagonal(torch.cdist(real_flatten, fake_flatten)) * (1/2) # 分母，这里计算了n²个距离，有性能优化空间
    print('L_GR', g_loss)

    # L_C / L_GC
    c_fake, c_fake_identity = C(fake)
    c_fake = torch.nn.functional.softmax(c_fake, dim=1) # 转 one-hot 概率向量
    c_fake_p = torch.gather(c_fake, 1, targets.view(batch, 1)) # 提出正确标签的概率 p
    c_fake_p = c_fake_p.view(batch)

    # c_loss = -log(C(x^')向量的第x^s的标签个元素) / 0.5*||fc(x^')-fc(x^s)||^2
    c_loss_a = - torch.log(c_fake_p) # 分子
    c_loss_b = torch.diagonal(torch.cdist(c_fake_identity, i_identity)) * (1/2) # 分母，这里计算了n²个距离，有性能优化空间
    c_loss = c_loss_a / c_loss_b
    print('L_C / L_GC', c_loss)

    # L_D / L_GD
    d_fake_p, d_fake_feature = D(fake)
    d_real_p, d_real_feature = D(real)
    d_fake_p = d_fake_p.view(batch) # 铺平
    d_real_p = d_real_p.view(batch)
    
    d_loss_a = - torch.log(d_real_p) - torch.log(1 - d_fake_p) # 分子
    d_loss_b = torch.diagonal(torch.cdist(d_fake_feature, d_real_feature)) * (1/2) # 分母，这里计算了n²个距离，有性能优化空间
    d_loss = d_loss_a / d_loss_b
    print('L_D / L_GD', d_loss)
  
    # TODO: KL距离约束 Attribute 不要和 Identity 分布太接近


    # i_optimizer.zero_grad()
    # i_loss.backward()
    # i_optimizer.step()
