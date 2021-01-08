# 读入、输出、转换 bvh 数据
import numpy as np
import torch
import re
import os

def save_bvh_to_file(filename, skeleton, frames):
  # skeleton: [93]
  # frames: [96, 240]

  skeleton = np.array(skeleton).reshape(-1)
  frames = np.array(frames).swapaxes(0, 1).reshape(-1)
  data = np.concatenate((skeleton, frames), axis=0)
  with open('./v5/template.bvh', 'r') as file:
    bvh = file.read().format(*data.tolist())
  with open(filename, 'w') as file:
    file.write(bvh)

def load_bvh_from_file(filename):
  with open(filename, 'r') as file:
    bvh = file.read()

  # 骨架信息 [93]
  skeleton = list(map(float, ' '.join(re.findall(r'OFFSET\s(.*?)\n\s*CHANNELS', bvh)).split(' ')))
  skeleton = np.array(skeleton)

  # 运动帧信息 [96, 240]
  bvh = bvh[bvh.find('Frame Time'):]
  bvh = bvh[bvh.find('\n') + 1:]
  bvh = bvh.strip()
  frames = list(map(lambda f : [float(x) for x in f.split(' ')], bvh.split('\n'))) # [240, 96]
  frames = np.array(frames).swapaxes(0, 1)

  # 身份标签 ID
  label = int(os.path.basename(filename).split('_')[0]) # 文件名第一个数字作为标签

  return (skeleton, frames, label)

def load_all_bvh_from_dirctory(dirpath):
  files = os.listdir(dirpath)
  files = filter(lambda f: f.split('.')[-1] == 'bvh', files) # 过滤出本目录所有 bvh 文件
  data = map(lambda f: load_bvh_from_file(os.path.join(dirpath, f)), files) # 载入每一个文件
  data = list(data)
  # data: [
  #   (skeleton: [93], frames: [96, 240], label: int),
  #   ...
  # ]
  return data

def transform_frames_to_detal_frames(frames):
  # frames: [96, 240]
  detal_frames = np.copy(frames)

  # 倒着往前减出变化量
  for i in range(detal_frames.shape[1] - 1, 0, -1):
    detal_frames[:, i] = detal_frames[:, i] - detal_frames[:, i - 1]

  # detal_frames: [
  #   原始动作帧,
  #   相对上一帧的变化量,
  #   相对上一帧的变化量,
  #   ...
  # ]
  return detal_frames # [96, 240]

def transform_detal_frames_to_frames(detal_frames):
  # detal_frame: [96, 240]
  frames = np.copy(detal_frames)

  # 正着往后加出原值
  for i in range(1, frames.shape[1]):
    frames[:, i] = frames[:, i] + frames[:, i - 1]

  # frames: [
  #   动作帧1,
  #   动作帧2,
  #   动作帧3,
  #   ...
  # ]
  return frames # [96, 240]

def get_data_frames_statistics(frames):
  # frames: [N, C, T]
  # 去掉了 t0 的基础帧

  data = np.concatenate(frames, axis=1) # 所有 bvh 文件按关节拼起来, [C, N * T]
  statistics = np.stack([data.min(axis=1), data.max(axis=1), data.mean(axis=1), data.std(axis=1)], axis=1)

  # statistics: [
  #   [min, max, mean, std], # 关节 0 相关的统计数据
  #   [min, max, mean, std], # 关节 1 相关的统计数据
  #   ...
  # ]
  return statistics # [C, 4]

def frames_to_normalized_frames(frames, statistics):
  # frames [C, T]
  normalized_frames = np.copy(frames)

  for i in range(normalized_frames.shape[0]): # 遍历关节通道
    c_min, c_max, c_mean, _ = statistics[i]
    if c_min == c_max: # 该通道最大最小一样，直接设置成 0，防止归一分母为零
      normalized_frames[i] = np.zeros_like(normalized_frames[i])
      continue
    # (x - mean) / (max - min)
    normalized_frames[i] = (normalized_frames[i] - c_mean) / (c_max - c_min)

  return normalized_frames

def normalized_frames_to_frames(normalized_frames, statistics):
  # normalized_frames [C, T]
  frames = np.copy(normalized_frames)

  for i in range(frames.shape[0]):
    c_min, c_max, c_mean, _ = statistics[i]
    # x * (max - min) + mean
    frames[i] = frames[i] * (c_max - c_min) + c_mean

  return frames


if __name__ == "__main__":
  data = load_all_bvh_from_dirctory('./v5/walk_id_compacted')
  data = data[0]
  skeleton, frames, label = data
  detal_frames = transform_frames_to_detal_frames(frames)
  print(detal_frames.shape)


  statistics = get_data_frames_statistics(detal_frames[:, 1:].reshape(1, 96, 239))

  print(detal_frames[:, 1:].max(), detal_frames[:, 1:].min())
  n_f = frames_to_normalized_frames(detal_frames[:, 1:], statistics)
  print(n_f.max(), n_f.min())
  s_f = normalized_frames_to_frames(n_f, statistics)
  print(s_f.max(), s_f.min())

  # frames_to_normalized_frames

  # data = [(skeleton, transform_frames_to_detal_frames(frames), label) for skeleton, frames, label in data]
  # statistics = get_data_frames_statistics(data)

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
