# 载入运动数据帧（240*96），忽略骨骼信息

import torch
import re
import os
import random

def loadBVHData(filename):
  with open(filename, 'r') as file:
    bvh = file.read()
    
    # 骨架信息
    skeleton = list(map(float, ' '.join(re.findall(r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
    skeleton.extend([0., 0., 0.]) # 补个零占位到 96，因为动作帧里开头有三个元素代表 xyz 坐标
    skeleton = [skeleton] # [1, 96]

    # 240 帧运动信息
    bvh = bvh[bvh.find('Frame Time'):]
    bvh = bvh[bvh.find('\n') + 1:]
    bvh = bvh.strip()
    frame = list(map(lambda f : [float(x) for x in f.split(' ')], bvh.split('\n'))) # [240, 96]

    # 全部信息
    data = torch.FloatTensor(skeleton + frame) # [1, 96] + [240, 96] = [241, 96]
    label = int(os.path.basename(filename).split('_')[0]) # 文件名第一个数字作为标签

    return (data, label)

def getAllData(dirpath):
  files = [f for f in os.listdir(dirpath)]
  data = []
  for file in files:
    filename = os.path.join(dirpath, file)
    data.append(loadBVHData(filename))
  return data


class MyDataset(torch.utils.data.Dataset):
  # rate 要使用的数据占全部的比率，暂时无用
  def __init__(self, dataset_dir='./v5/walk_id_compacted', rate=1.0):
    self.data = getAllData(dataset_dir)
    # random.shuffle(self.data)
    # self.data = self.data[:int(len(self.data) * rate)]
    # exit()

    self.combination = [] # 不同人物的运动数据杂交（笛卡尔积）
    indexes = range(len(self.data))
    for a in indexes:
      for b in indexes:
        _, label_a = self.data[a]
        _, label_b = self.data[b]
        if (label_a == label_b): # 不要自交
          continue
        self.combination.append((a, b))
    print('数据及杂交后数量', len(self.combination))

    # 因为 id 从零开始，分类总数直接取最后一个文件的 id + 1
    self.category_num = int(os.listdir(dataset_dir)[-1].split('_')[0]) + 1
    
  def __getitem__(self, index):
    index_a, index_b = self.combination[index]
    data_a, label_a = self.data[index_a]
    data_b, label_b = self.data[index_b]
    return data_a, label_a, data_b, label_b

  def __len__(self):
    return len(self.combination)


if __name__ == "__main__":
  # data = getAllData('./v5/walk_id_compacted')
  # print(len(data))
  
  # data = loadBVHData('./v5/walk/02_01_1.bvh')
  # print(data)
  
  # print(data)
  # print('[调试]', 'dataloader.py')
  ds = MyDataset()
  print(ds.category_num)
  # dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1, shuffle=True)

  # for x in dl:
    # print(x[0].shape)

