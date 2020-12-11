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


if __name__ == "__main__":
  data = getAllData('./v5/walk_id_compacted')
  ## TODO：我觉得，应该每个关节的240帧分别归一化，因为每个关节的活动范围各有不同
  ## TODO：晚点实现

  # data = torch.cat(tuple(map(lambda x: x[0].view(-1), data)), 0) # 所有数据铺平成一维
  # print(data.shape)
  # print(data.min(), data.max())
  # print(data.mean(), data.std())

    # print(x.view(-1)[1000:].max(), x.view(-1).min())
  
