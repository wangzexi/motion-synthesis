## 运动数据解析出的格式

## 文档

【腾讯文档】运动数据格式
https://docs.qq.com/sheet/DZmhrcnZXcnhiS1dG

【腾讯文档】网络结构
https://docs.qq.com/slide/DZkFiUXZiQlF0dnV3

## 改动

- 骨骼 与 第一帧 不再参与训练
- 输出 bvh 文件时，x_a 的 骨骼 与 第一帧 会直接装载到 G 生成的 增量帧 上

- 参与训练的 帧数据 均处理为相对上一帧的 增量
- 仍然会对所有 增量 进行归一化

- 不再训练 C，直接使用 I 替代
