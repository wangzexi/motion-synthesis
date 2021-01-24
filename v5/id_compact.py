'''
本脚本可以把一个目录下的 bvh 文件们，稀疏的文件名的 id 压实、压连续
'''

import os

i = 0
mapper = {}
def run(dirpath):
  files = [f for f in os.listdir(dirpath)]
  for file in files:
    old_id = int(file.split('_')[0])
    
    new_id = -1
    if old_id not in mapper:
      global i
      mapper[old_id] = i
      new_id = i
      i = i + 1
    else:
      new_id = mapper.get(old_id)

    new_file = str(new_id).zfill(2) + file[file.find('_'):]

    old_file_path = os.path.join(dirpath, file)
    new_file_path = os.path.join(dirpath, new_file)

    print(old_file_path, new_file_path)
    os.rename(old_file_path, new_file_path)

if __name__ == "__main__":
  run('./walk_id_compacted')
