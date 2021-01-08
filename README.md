## 运动数据解析出的格式

# 数据格式

skeleton: [93]
frames: [96, 240]

## 文档

以下文档还未更新：

【腾讯文档】运动数据格式
https://docs.qq.com/sheet/DZmhrcnZXcnhiS1dG

【腾讯文档】网络结构
https://docs.qq.com/slide/DZkFiUXZiQlF0dnV3


## 其它

TCN 里每层之间都有短接和非线性变换。

bvh -> frames:[96, 240] -> detal_frames:[96, 240] -> statistics detal_frames(without t0) -> normalized detal_frames(with t0)

待尝试：A 在 C 维上卷，I 在 T 维上卷。

## 改动

### 最新

- 生成器改为 TCN
- 修正为在时间维度上卷积
- 分离 I 的训练过程，进行预训练防止过拟合的 I 产生的糟糕的 Loss_GC 反馈

### 历史

- 骨骼 与 第一帧 不再参与训练
- 输出 bvh 文件时，x_a 的 骨骼 与 第一帧 会直接装载到 G 生成的 增量帧 上
- 参与训练的 帧数据 均处理为相对上一帧的 增量
- 仍然会对所有 增量 进行归一化
- 不再训练 C，直接使用 I 替代
