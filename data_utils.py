# 载入运动数据帧（240*96），忽略骨骼信息

import torch
import re
import os
import random
import numpy as np

def save_bvh_to_file(filename, data):
  data = np.array(data)
  # data [241, 96]
  with open('./v5/template.bvh', 'r') as file:
    bvh = file.read().format(*data.reshape(-1).tolist())
  with open(filename, 'w') as file:
    file.write(bvh)

def load_bvh_from_file(filename):
  with open(filename, 'r') as file:
    bvh = file.read()
    
  # 骨架信息
  skeleton = list(map(float, ' '.join(re.findall(r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
  skeleton.extend([0., 0., 0.]) # 尾部补个零占位到 96，因为动作帧里开头有三个元素代表 xyz 坐标
  skeleton = [skeleton] # [1, 96]

  # 240 帧运动信息
  bvh = bvh[bvh.find('Frame Time'):]
  bvh = bvh[bvh.find('\n') + 1:]
  bvh = bvh.strip()
  frame = list(map(lambda f : [float(x) for x in f.split(' ')], bvh.split('\n'))) # [240, 96]

  # 全部信息
  data = skeleton + frame # [1, 96] + [240, 96] = [241, 96]
  label = int(os.path.basename(filename).split('_')[0]) # 文件名第一个数字作为标签

  return (data, label)

def load_all_bvh_from_dirctory(dirpath):
  files = os.listdir(dirpath)
  files = filter(lambda f: f.split('.')[-1] == 'bvh', files) # 过滤出本目录所有bvh文件
  data = map(lambda f: load_bvh_from_file(os.path.join(dirpath, f)), files) # 载入每一个文件
  data = list(data)
  return data

def get_statistics(data):
  # data [161, 241, 96]
  data = np.array(data)

  ## 对骨架和每个关节分别统计，因为每个关节的活动范围各有不同
  s_data = data[:, 0, :].reshape(-1) # 所有的骨长偏移都串进来
  s_statistics = [s_data.min(), s_data.max(), s_data.mean(), s_data.std()]
  # print(s_statistics)

  j_data = data[:, 1:, :].swapaxes(1,2) # [161, 96, 240]
  j_data = np.concatenate(j_data, axis=1) # 所有 bvh 文件按关节拼起来
  j_statistics = np.stack([j_data.min(axis=1), j_data.max(axis=1), j_data.mean(axis=1), j_data.std(axis=1)], axis=1)
  # j_statistics[7][1] 意味着 7 号关节的 max 值
  # print(j_statistics[0])
  # print(j_statistics.shape)

  statistics = np.concatenate(([s_statistics], j_statistics), axis=0)

  # 0号是骨架的min、max、mean、std
  # 1号是1号关节的min、max、mean、std
  # ...
  # 96号是96号关节的min、max、mean、std
  # print(statistics)
  # print(statistics.shape)
  return statistics

def normalized(data, statistics):
  # data [241, 96]
  data = np.array(data)
  
  # (x - mean) / (max - min)
  data[0, :] = (data[0, :] - statistics[0, 2]) / (statistics[0, 1] - statistics[0, 0])

  for i in range(data.shape[1]):
    if statistics[i + 1, 0] == statistics[i + 1, 1]: # 最大最小一样，直接设置成0
      data[1:, i] = np.zeros_like(data[1:, i])
      continue
    data[1:, i] = (data[1:, i] - statistics[i + 1, 2]) / (statistics[i + 1, 1] - statistics[i + 1, 0])

  return data.tolist()

def denormalized(data, statistics):
  # data [241, 96]
  data = np.array(data)
  
  # x * (max - min) + mean
  data[0, :] = data[0, :] * (statistics[0, 1] - statistics[0, 0]) + statistics[0, 2]

  for i in range(data.shape[1]):
    data[1:, i] = data[1:, i] * (statistics[i + 1, 1] - statistics[i + 1, 0]) + statistics[i + 1, 2]

  return data.tolist()


if __name__ == "__main__":
  data = load_all_bvh_from_dirctory('./v5/walk_id_compacted')
  # statistics = get_statistics(data)
  # np.savetxt('./v5/walk_id_compacted/_min_max_mean_std.csv', statistics)
  
  data_without_label = [x[0] for x in data] # 去除标签
  statistics = np.loadtxt('./v5/walk_id_compacted/_min_max_mean_std.csv')

  save_bvh_to_file('./output.bvh', data[-1][0])
  # sss = data[-1][0]
  # aaa = normalized(sss, statistics)
  # bbb = denormalized(aaa, statistics)
  # print(sss)
  # print(bbb)

  # exit()
  # data = [(normalized(x[0], statistics), x[1]) for x in data]
  # print(data[-1])

  # ddd = json.loads(sss)
  # with open('./v5/walk_id_compacted/min_max_mean_std.json', 'w') as file:
    # file.write(json.dumps(statistics))

  # skeletons = data[:, 0, :].reshape(-1) # 所有的骨长偏移都串进来
  # joints = [list() for _ in range(data.shape[2])] # 关节个数，所有的关节分门别类串进去

  # for x in data:
  #   skeletons.append(x[0])
  #   frames = x[1:]
  #   for i in range(frames.shape[1]): # 对每个关节归一
  #     joints[i].append(frames[:, i])

  # # 转换成 tensor
  # skeletons = np.vstack(skeletons)
  # joints = list(map(lambda x: np.vstack(x), joints)) # TODO: 本质上是个维度交换，使用numpy重写
  # # print(skeletons.min(), skeletons.max(), skeletons.mean(), skeletons.std())
  # print(joints[0].shape)
  # exit()

  # normalization = []
  # # 对骨长求参数
  # normalization.append((skeletons.min().item(), skeletons.max().item(), skeletons.mean().item(), skeletons.std().item()))
  # # 对每个关节求参数
  # for joint in joints:
  #   normalization.append((joint.min().item(), joint.max().item(), joint.mean().item(), joint.std().item()))

  # sss = json.dumps(normalization)
  # # ddd = json.loads(sss)
  # with open('./v5/walk_id_compacted/min_max_mean_std.json', 'w') as file:
  #   file.write(sss)
