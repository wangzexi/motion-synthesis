# 读入和输出 bvh 文件

import torch
import re
import os
import random
import numpy as np

def save_bvh_to_file(filename, data):
  # data: [
  #   skeleton,
  #   frames
  # ]
  data = np.array(data) # [241, 96]
  with open('./v5/template.bvh', 'r') as file:
    bvh = file.read().format(*data.reshape(-1).tolist())
  with open(filename, 'w') as file:
    file.write(bvh)

def load_bvh_from_file(filename):
  with open(filename, 'r') as file:
    bvh = file.read()
    
  # 骨架信息 [96]
  skeleton = list(map(float, ' '.join(re.findall(r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
  skeleton.extend([0., 0., 0.]) # 尾部补个零占位到 96，因为动作帧里开头有三个元素代表 xyz 坐标
  # 虽然骨骼分离了，但 bvh 模板文件里骨骼留了是 96 个空位，所以依然需要补 0 个零
  skeleton = np.array(skeleton)

  # 运动帧信息 [240, 96]
  bvh = bvh[bvh.find('Frame Time'):]
  bvh = bvh[bvh.find('\n') + 1:]
  bvh = bvh.strip()
  frames = list(map(lambda f : [float(x) for x in f.split(' ')], bvh.split('\n'))) # [240, 96]
  frames = np.array(frames)

  # 身份标签 ID
  label = int(os.path.basename(filename).split('_')[0]) # 文件名第一个数字作为标签

  return (skeleton, frames, label)

def load_all_bvh_from_dirctory(dirpath):
  files = os.listdir(dirpath)
  files = filter(lambda f: f.split('.')[-1] == 'bvh', files) # 过滤出本目录所有 bvh 文件
  data = map(lambda f: load_bvh_from_file(os.path.join(dirpath, f)), files) # 载入每一个文件
  data = list(data)
  # data: [
  #   (skeleton: [96], frames: [240, 96], label: int),
  #   ...
  # ]
  return data

def transform_frames_to_detal_frames(frames):
  # frame: [240, 96]
  detal_frames = np.copy(frames)

  # 倒着往前减出变化量
  for i in range(detal_frames.shape[0] - 1, 0, -1):
    detal_frames[i] = detal_frames[i] - detal_frames[i - 1]

  # detal_frames: [
  #   原始动作帧,
  #   相对上一帧的变化量,
  #   相对上一帧的变化量,
  #   ...
  # ]
  return detal_frames # [240, 96]

def transform_detal_frames_to_frames(detal_frames):
  # detal_frame: [240, 96]
  frames = np.copy(detal_frames)

  # 正着往后加出原值
  for i in range(1, frames.shape[0]):
    frames[i] = frames[i] + frames[i - 1]

  # frames: [
  #   动作帧1,
  #   动作帧2,
  #   动作帧3,
  #   ...
  # ]
  return frames # [240, 96]

def get_data_frames_statistics(frames):
  # frames: [161, 239, 96]

  j_data = frames.swapaxes(1, 2) # [161, 96, 239]
  j_data = np.concatenate(j_data, axis=1) # 所有 bvh 文件按关节拼起来, [96, 161 * 239]
  j_statistics = np.stack([j_data.min(axis=1), j_data.max(axis=1), j_data.mean(axis=1), j_data.std(axis=1)], axis=1)

  # j_statistics: [
  #   [min, max, mean, std], # 关节 0 相关的统计数据
  #   [min, max, mean, std], # 关节 1 相关的统计数据
  #   ...
  # ]
  return j_statistics # [96, 4]

def frames_to_normalized_frames(frames, statistics):
  # frames [240, 96]
  normalized_frames = np.copy(frames)

  for i in range(normalized_frames.shape[1]):
    if statistics[i, 0] == statistics[i, 1]: # 最大最小一样，直接设置成 0，防止归一分母为零
      normalized_frames[:, i] = np.zeros_like(normalized_frames[:, i])
      continue
    # (x - mean) / (max - min)
    normalized_frames[:, i] = (normalized_frames[:, i] - statistics[i, 2]) / (statistics[i, 1] - statistics[i, 0])

  return normalized_frames

def normalized_frames_to_frames(normalized_frames, statistics):
  # normalized_frames [240, 96]
  frames = np.copy(normalized_frames)

  for i in range(frames.shape[1]):
    # x * (max - min) + mean
    frames[:, i] = frames[:, i] * (statistics[i, 1] - statistics[i, 0]) + statistics[i, 2]

  return frames


if __name__ == "__main__":
  data = load_all_bvh_from_dirctory('./v5/walk_id_compacted')

  data = [(skeleton, transform_frames_to_detal_frames(frames), label) for skeleton, frames, label in data]
  statistics = get_data_frames_statistics(data)

  # frames = data[0][1]

  # normalized_frames = frames_to_normalized_frames(frames, statistics)
  # new_frames = normalized_frames_to_frames(normalized_frames, statistics)
  # print(normalized_frames.min(), normalized_frames.max())
  # print((new_frames - frames).max())


  # print(statistics.shape)

  # np.savetxt('./v5/walk_id_compacted/_min_max_mean_std.csv', statistics)
  
  # data_without_label = [x[0] for x in data] # 去除标签
  # statistics = np.loadtxt('./v5/walk_id_compacted/_min_max_mean_std.csv')

  # save_bvh_to_file('./output.bvh', data[-1][0])
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
