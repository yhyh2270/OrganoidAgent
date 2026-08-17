# Yichao 数据集结构说明：明场到荧光的 Pix2pix 任务

英文版见：

- `references/Yichao/README.md`

本文档整理以下目录中的当前 Yichao 数据布局：

- v1 根目录：`/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-1`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-2`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-3`
- `/home/lachlan/ProjectsLFS/OrganoidAgent/Data-Yichao-v1/Data-Yichao-4`
- v2 根目录：`/home/lachlan/ProjectsLFS/OrganoidAgent/DATA-Yichao-v2`

本仓库当前采用的监督映射是：

- 输入：`c1` 明场
- 目标：`c0` 荧光

这个假设已经被以下脚本使用：

- `BioAgentUtils/prepare_yichao_pairs_to_npy.py`
- `BioAgentUtils/train_pix2pix_yichao.py`


## 当前拆分方式

现在的拆分方式是 **先按原始 LIF 文件拆分**，不是按 day 拆分。

当前目录含义：

- `Data-Yichao-3` = `N39_TriRep_DF.lif` 对应的数据
- `Data-Yichao-4` = `N39_TriRep_DF_2.lif` 对应的数据

当前磁盘状态：

- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_position`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all_by_position`

重要说明：

- `N39_TriRep_DF.lif` 已成功导出
- `N39_TriRep_DF_2.lif` 也已成功导出
- `*_by_position` 目录是对同一批 JPEG 的重组视图，按被监测的 position 进行整理


## 可用数据的基本单位

对于 pix2pix，最合适的监督样本定义是：

- 在固定 `series/position`、`z`、`t` 下的一对 2D 图像
- 即明场 `c1` 与荧光 `c0` 的同位配对

对一个 LIF series 而言：

- 可用配对样本数 = `z_count * time_count`
- 导出的 JPEG 总数 = `z_count * time_count * 2`

因为每个 `(t, z)` 平面都会导出两个通道。


## 重要的重叠警告

`Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF.lif` 不能被视为独立评估集。

其中 5 个静态 MUC2 样本，与下面文件中的前 5 个静态样本是字节级完全一致的：

- `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF.lif`

因此：

- 用 `Data-Yichao-2` 训练，再用 `Data-Yichao-1` 测试，会发生数据泄漏
- 当前仓库默认配置适合快速 smoke test，但不适合做严谨的最终 benchmark


## 各数据目录的结构说明

### Data-Yichao-1

LIF 文件：

- `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF.lif`

内容：

- 共 5 个 series
- 全部是静态单平面样本
- 名称为：
  - `N39_TriRep_MUC2_mNeon_20X_1`
  - `N39_TriRep_MUC2_mNeon_20X_2`
  - `N39_TriRep_MUC2_mNeon_20X_3`
  - `N39_TriReP_MUC2_mNeon_20X_4`
  - `N39_TriRep_MUC2_mNeon_20X_5`

采集结构：

- XY 尺寸：`1024 x 1024`
- 通道数：`2`
- 每个样本的 z 深度：`1`
- 每个样本的时间点数：`1`
- 每个 series 的可用配对数：`1`
- 总可用配对数：`5`
- 近似像素尺寸：`0.303 um/pixel`

解释：

- 这是 5 个独立静态视野
- 没有 z-stack
- 没有 time-lapse
- 这 5 个样本全部在 `Data-Yichao-2` 中重复出现


### Data-Yichao-2

LIF 文件：

- `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF.lif`

内容：

- 共 11 个 series
- 其中 5 个是来自 `Data-Yichao-1` 的静态 MUC2 series
- 另外 6 个是 Day-2 的动态 position

series 结构如下：

| Series 组别 | 数量 | XY | Z | T | 可用配对数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静态 MUC2 | 5 | 1024x1024 | 1 | 1 | 5 |
| `N39_TriRep_DF_D2/Position001` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position002` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position003` | 1 | 1024x1024 | 11 | 16 | 176 |
| `N39_TriRep_DF_D2/Position004` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position005` | 1 | 1024x1024 | 9 | 16 | 144 |
| `N39_TriRep_DF_D2/Position006` | 1 | 1024x1024 | 11 | 16 | 176 |

动态 Day-2 部分的采集结构：

- 通道数：`2`
- 近似像素尺寸：`0.568 um/pixel`
- z 步长绝对值：约 `2.469 um`
- 时间步长：约 `3622 s`，即 `60.4 min`

统计：

- 包含静态重复图像时，总可用配对数：`965`
- 扣除 Y1 重复后，唯一动态配对数：`960`

解释：

- 这是一个混合型文件
- 同时包含静态 MUC2 图像与动态 Day-2 stack
- 对训练真正新增的内容，主要是这 6 个 Day-2 动态 position


### Data-Yichao-3

当前角色：

- 第一个按 LIF 文件拆分后的目录
- 只包含 `N39_TriRep_DF.lif` 对应的数据

当前文件：

- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF_jpeg_all_by_position`

当前导出总量：

- 可用配对数：`10019`
- `N39_TriRep_DF_jpeg_all` 中 JPEG 总数：`20038`
- `N39_TriRep_DF_jpeg_all_by_position` 中 position 文件夹数：`13`

series 明细：

| Series | XY | Z | T | 可用配对数 |
| --- | ---: | ---: | ---: | ---: |
| `Experiment_1 Day_2/Position001` | 512x512 | 26 | 41 | 1066 |
| `Experiment_1 Day_2/Position002` | 512x512 | 10 | 41 | 410 |
| `Experiment_1 Day_2/Position003` | 512x512 | 20 | 41 | 820 |
| `Experiment_1 Day_3/Position001` | 512x512 | 16 | 49 | 784 |
| `Experiment_1 Day_3/Position002` | 512x512 | 24 | 49 | 1176 |
| `Experiment_1 Day_3/Position003` | 512x512 | 8 | 49 | 392 |
| `Experiment_1 Day_3/Position004` | 512x512 | 11 | 49 | 539 |
| `Experiment_1 Day_3/Position005` | 512x512 | 32 | 49 | 1568 |
| `Experiment_1 Day_4/Position001` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position002` | 512x512 | 25 | 32 | 800 |
| `Experiment_1 Day_4/Position003` | 512x512 | 18 | 32 | 576 |
| `Experiment_1 Day_4/Position004` | 512x512 | 23 | 32 | 736 |
| `Experiment_1 Day_4/Position005` | 512x512 | 18 | 32 | 576 |

按天汇总：

| Day | Position 数 | 可用配对数 |
| --- | ---: | ---: |
| Day 2 | 3 | 2296 |
| Day 3 | 5 | 4459 |
| Day 4 | 5 | 3264 |

采集结构：

- 通道数：`2`
- 近似像素尺寸：`1.137 um/pixel`
- Day 2 z 步长：约 `2.000 um`
- Day 3 z 步长：约 `1.608 um`
- Day 4 z 步长：约 `2.000 um`
- 时间步长：约 `1800-1805 s`，即约 `30 min`

解释：

- 这是主要的动态 monitoring 数据集
- 它包含 Day 2、Day 3、Day 4 的 time-lapse z-stack 数据


### Data-Yichao-4

当前角色：

- 第二个按 LIF 文件拆分后的目录
- 只包含 `N39_TriRep_DF_2.lif` 对应的数据

当前文件：

- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all`
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2_jpeg_all_by_position`

当前导出总量：

- 可用配对数：`7940`
- `N39_TriRep_DF_2_jpeg_all` 中 JPEG 总数：`15880`
- `N39_TriRep_DF_2_jpeg_all_by_position` 中 position 文件夹数：`9`

series 明细：

| Series | XY | Z | T | 可用配对数 |
| --- | ---: | ---: | ---: | ---: |
| `Experiment_1 Day_2/Position001` | 512x512 | 16 | 49 | 784 |
| `Experiment_1 Day_2/Position002` | 512x512 | 22 | 49 | 1078 |
| `Experiment_1 Day_2/Position003` | 512x512 | 22 | 49 | 1078 |
| `Experiment_1 Day_2/Position004` | 512x512 | 20 | 49 | 980 |
| `Experiment_1 Day_2/Position005` | 512x512 | 27 | 49 | 1323 |
| `Experiment_1 Day_3/Position001` | 512x512 | 38 | 31 | 1178 |
| `Experiment_1 Day_3/Position003` | 512x512 | 23 | 31 | 713 |
| `Experiment_1 Day_3/Position004` | 512x512 | 17 | 31 | 527 |
| `Experiment_1 Day_3/Position005` | 512x512 | 9 | 31 | 279 |

按天汇总：

| Day | Position 数 | 可用配对数 |
| --- | ---: | ---: |
| Day 2 | 5 | 5243 |
| Day 3 | 4 | 2697 |

解释：

- 这个目录已经按 LIF 文件正确拆分
- 它包含第二个动态 monitoring 数据集
- 包含 Day 2 与 Day 3 的 time-lapse z-stack


## 这些文件夹到底表示什么

### `N39_TriRep_DF_jpeg_all`

这是“平铺式”导出目录。

每个文件都是一个 2D 平面，对应某个 position、某个时间点、某个 z 深度和某个通道。

例如：

```text
00_Experiment_1_Day_2_Position001_t040_z025_c1.jpg
```

表示：

- `00`：LIF 导出中的 series 序号
- `Experiment_1_Day_2_Position001`：一个被监测的 position
- `t040`：第 40 个时间点
- `z025`：第 25 个 z 平面
- `c1`：第 1 个通道，这里视为荧光


### `N39_TriRep_DF_jpeg_all_by_position`

这是与上面相同的数据，但按 LIF series 名称重新分组。

这里的 “object” 更适合理解为：

- 一个成像 position
- 一个固定视野
- 一个被持续监测的 sample stack

也就是说，`*_by_position` 下的一个子目录就是一个被持续监测的 position。


### `Data-Yichao-v1/Data-Yichao-2/P11N&N39_Rep_DF_jpeg_all_by_object`

这个目录里混有两种不同类型的数据：

- 动态 Day-2 monitoring position：`N39_TriRep_DF_D2_Position001..006`
- 静态单次成像的 MUC2 样本：`N39_TriRep_MUC2_mNeon_20X_1..5`

所以其中并不是每个子目录都是 time-lapse 序列。


### `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg`

这是较早期的旧版导出目录。

它的特点是：

- 文件名没有显式写出 `t000_z000`
- 但本质上仍然只是 5 个静态样本的单次 c0/c1 配对

更规范、信息更完整的等价导出目录是：

- `Data-Yichao-v1/Data-Yichao-1/P11N&N39_Rep_DF_jpeg_all`


## 基于底层原始文件的唯一可用数据

去除重复内容后：

- `Data-Yichao-1`：`5` 个配对样本，但全部在 `Data-Yichao-2` 中重复
- `Data-Yichao-2`：`960` 个唯一动态配对样本
- `Data-Yichao-v1/Data-Yichao-3/N39_TriRep_DF.lif`：`10019` 个唯一动态配对样本
- `Data-Yichao-v1/Data-Yichao-4/N39_TriRep_DF_2.lif`：`7940` 个动态配对样本

因此，当前非空且可用的原始 LIF 来源中的唯一配对平面总数为：

- `18924`

这些可用原始来源中的唯一 series/position 数量为：

- 5 个静态 MUC2 series
- 6 个 Yichao-2 Day-2 动态 position
- `N39_TriRep_DF.lif` 中的 13 个动态 position
- `N39_TriRep_DF_2.lif` 中的 9 个动态 position
- 总计 `33`


## 这里的“重复”或“replication”该如何理解

文件名中虽然出现了 `TriRep`，但当前检查到的 LIF 元数据并不能可靠给出严格的生物学 replicate 标签。

更稳妥的解释是：

- 一个 LIF series 就是一个视野 / 一个 position / 一个 sample stack
- 同一天内不同 position 应视为不同样本
- z 平面是同一样本在不同深度的重复观测
- 时间点是同一样本在不同时刻的重复观测
- 不要直接把 `TriRep` 当成可用于评估划分的机器可解析 replicate 标签


## 对 Pix2pix 训练的含义

推荐的监督样本定义：

- 一个固定 `position`、`t`、`z` 下的 `c0 -> c1` 配对平面

推荐的划分规则：

- 按 position 切分，而不是随机按单张平面切分

原因：

- 相邻 z 平面高度相关
- 相邻时间点高度相关
- 随机打散到 train/val/test 会造成严重信息泄漏


### 实用的第一版基线

当前最干净的基线做法是：

- 先在 `Data-Yichao-3` 上训练
- 在 `N39_TriRep_DF_2.lif` 被恢复前，把 `Data-Yichao-4` 视为暂不可用


### 是否应该把 Yichao-2 和 Data-Yichao-3 混合训练

不建议一开始就直接混合。

Yichao-2 的动态数据与 `N39_TriRep_DF.lif` 的空间采样不同：

- Yichao-2 动态部分：`1024 x 1024`，约 `0.568 um/pixel`
- `N39_TriRep_DF.lif`：`512 x 512`，约 `1.137 um/pixel`

这意味着：

- 物理视野与有效尺度不同
- 不做归一化就直接混合，会引入明显的 domain shift

更好的做法是：

- 先在 `Data-Yichao-3` 上建立干净基线
- 再考虑在统一物理尺度后加入 Yichao-2 的动态 position


## 总结

如果目标是学习明场到荧光的 pix2pix 映射：

- `Data-Yichao-1` 只有静态数据，而且不适合作为独立测试集
- `Data-Yichao-2` 包含有价值的动态 Day-2 数据，但也混有重复静态内容
- `Data-Yichao-3` 现在是第一个按 LIF 拆分后的目录，包含 `N39_TriRep_DF.lif` 的完整导出结果
- `Data-Yichao-4` 现在是第二个按 LIF 拆分后的目录，包含 `N39_TriRep_DF_2.lif` 的导出结果
