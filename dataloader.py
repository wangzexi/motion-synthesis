# 载入运动数据帧（240*96），忽略骨骼信息

import torch
import os
import numpy as np
import data_utils

class MyDataset(torch.utils.data.Dataset):
  # rate 要使用的数据占全部的比率，暂时无用
  def __init__(self, dataset_dir='./v5/walk_id_compacted'):
    self.data = data_utils.load_all_bvh_from_dirctory(dataset_dir)
    
    # 计算 min max mean std
    data_without_lable = [x[0] for x in self.data]
    self.statistics = data_utils.get_statistics(data_without_lable)
    np.savetxt('./v5/walk_id_compacted/_min_max_mean_std.csv', self.statistics)
    
    # 归一化
    self.data = [(torch.FloatTensor(data_utils.normalized(x[0], self.statistics)), x[1]) for x in self.data]

    # 不同人物的运动数据杂交（笛卡尔积）
    self.combination = []
    indexes = range(len(self.data))
    for a in indexes:
      for b in indexes:
        _, label_a = self.data[a]
        _, label_b = self.data[b]
        if (label_a == label_b): # 不要自交
          continue
        self.combination.append((a, b))
    print('数据集杂交后数量', len(self.combination))

    # 因为 id 从零开始，分类总数直接取最后一个文件的 id + 1
    files = list(filter(lambda f: f.split('.')[-1] == 'bvh', os.listdir(dataset_dir)))
    self.category_num = int(files[-1].split('_')[0]) + 1

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

