import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import EncoderTCN
from dataloader import get_i_train_and_test_dataset
import data_utils
import pathlib

torch.autograd.set_detect_anomaly(True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set, test_set = get_i_train_and_test_dataset(dataset_dir='./v5/walk_id_compacted')

batch_size = 20
learning_rate = 1e-4
epochs_num = 300
category_num = train_set.category_num

train_dataloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=False)

# 预训练 identity 分类器
I = EncoderTCN(
  in_channel_num=96, # 96 个关节通道
  identity_dim=64,
  category_num=category_num,
  level_channel_num=256,
  level_num=7,
  kernel_size=3,
  dropout=0.3
).to(device)

I_path = './models/I.pt'
# torch.save(I.state_dict(), I_path) # 保存
# I.load_state_dict(torch.load(I_path)) # 载入

# if os.path.isfile(i_path): # 如果有预训练的 I，就直接载入，跳过训练
#   I.load_state_dict(torch.load(I_path))
# else:

cross_entropy_loss = torch.nn.CrossEntropyLoss()
i_optimizer = torch.optim.Adam(I.parameters(), learning_rate)

accuracy_train = []
accuracy_test = []

for epoch in range(epochs_num):
  # 训练
  I.train()
  correct_num = 0
  total_num = 0
  for batch_idx, (_, frames, label) in enumerate(train_dataloader):
    frames = frames[:, :, 1:].to(device=device)
    label = label.to(device=device)

    ctg, _ = I(frames)
    Loss_ctg = cross_entropy_loss(ctg, label) # 分类损失
    print('Loss_ctg', Loss_ctg.detach().cpu().item())

    i_optimizer.zero_grad()
    Loss_ctg.backward()
    i_optimizer.step()

    pred_label = torch.argmax(ctg, 1)
    correct_num += (pred_label == label).sum().float()
    total_num += len(label)

  accuracy_train.append((correct_num / total_num).detach().cpu().item())

  # 测试
  I.eval()
  correct_num = 0
  total_num = 0
  for batch_idx, (_, frames, label) in enumerate(test_dataloader):
    frames = frames[:, :, 1:].to(device=device)
    label = label.to(device=device)
    
    ctg, _ = I(frames)

    correct_num += (torch.argmax(ctg, 1) == label).sum().float()
    total_num += len(label)

  accuracy_test.append((correct_num/total_num).detach().cpu().item())

  # 画损失图
  plt.plot(np.arange(len(accuracy_train)), accuracy_train, 'orange')
  plt.plot(np.arange(len(accuracy_test)), accuracy_test, 'green')
  plt.savefig('losses.png')
  plt.close()

  print('轮次 {} 批次 {}'.format(epoch, batch_idx))
  torch.save(I.state_dict(), I_path)
