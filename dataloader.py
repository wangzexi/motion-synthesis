# 载入运动数据帧（240*96），忽略骨骼信息

import torch
import os
import numpy as np
import data_utils
import random

def load_dir_data_statistics_category_num(dataset_dir='./v5/walk_id_compacted'):
  data = data_utils.load_all_bvh_from_dirctory(dataset_dir)

  # 将帧数据转为增量形式
  data = [(skeleton, data_utils.transform_frames_to_detal_frames(frames), label) for skeleton, frames, label in data]

  # 计算统计数据 min, max, mean, std
  all_frames = np.array([x[1] for x in data]) # [N, C, 240]
  all_frames = all_frames[:, :, 1:] # [N, C, 239] 统计时去除第一帧基础，因为它不是增量
  statistics = data_utils.get_data_frames_statistics(all_frames)
  np.savetxt(os.path.join(dataset_dir, '_min_max_mean_std.csv'), statistics)
  
  # 将帧数据归一化
  data = [(skeleton, torch.FloatTensor(data_utils.frames_to_normalized_frames(frames, statistics)), label) for skeleton, frames, label in data]

  # 因为 id 从零开始，分类总数直接取最后一个 bvh 文件的 id + 1
  files = list(filter(lambda f: f.split('.')[-1] == 'bvh', sorted(os.listdir(dataset_dir))))
  category_num = int(files[-1].split('_')[0]) + 1

  return (data, statistics, category_num)

# 基础数据集包装器
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, data, statistics, category_num):
    self.data = data
    self.statistics = statistics
    self.category_num = category_num

  def __getitem__(self, index):
    return self.data[index]

  def __len__(self):
    return len(self.data)

# 杂交数据集包装器
class Hybrid_Dataset(MyDataset):
  def __init__(self, data, statistics, category_num, drop_selfing=True): # 是否丢弃自交
    self.data = data
    self.statistics = statistics
    self.category_num = category_num

    # 不同人物的运动数据杂交（笛卡尔积）
    self.combination = []
    indexes = range(len(self.data))
    for a in indexes:
      for b in indexes:
        _, _, label_a = self.data[a]
        _, _, label_b = self.data[b]
        if (drop_selfing and label_a == label_b): # 不要自交
          continue
        self.combination.append((a, b))
    print('数据集杂交后数量', len(self.combination))

  def __getitem__(self, index):
    index_a, index_b = self.combination[index]
    return (self.data[index_a], self.data[index_b])

  def __len__(self):
    return len(self.combination)

# 返回用于预训练 I 的训练集、测试集
def get_i_train_and_test_dataset(dataset_dir='./v5/walk_id_compacted'):
  data, statistics, category_num = load_dir_data_statistics_category_num(dataset_dir)
  random.shuffle(data)

  # 七成用于训练、三成用于测试
  train_set = data[0:int(len(data) * 0.7) + 1]
  test_set = data[:int(len(data) * 0.3)]

  return (
    MyDataset(train_set, statistics, category_num),
    MyDataset(test_set, statistics, category_num)
  )

if __name__ == "__main__":
  train, test = get_i_train_and_test_dataset()
  print(train[0])

