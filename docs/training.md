# 训练指南

本文档详细介绍如何训练ProPainter模型，包括数据准备、配置设置和训练流程。

## 训练概述

ProPainter包含两个主要组件，需要分别训练：
1. **循环光流补全网络** (Recurrent Flow Completion Network)
2. **ProPainter主网络** (ProPainter Main Network)

## 数据集准备

### 支持的数据集

| 数据集 | 用途 | 视频数量 | 下载链接 |
|--------|------|----------|----------|
| YouTube-VOS | 训练+评估 | 3,471 (训练) + 508 (测试) | [官方链接](https://competitions.codalab.org/competitions/19544) |
| DAVIS | 评估 | 50 (来自90个视频) | [官方链接](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip) |

### 数据结构

训练前需要将数据组织成以下结构：

```
datasets/
├── youtube-vos/
│   ├── JPEGImages_432_240/
│   │   ├── video1/
│   │   │   ├── 00000.jpg
│   │   │   ├── 00001.jpg
│   │   │   └── ...
│   │   └── video2/
│   ├── test_masks/
│   │   ├── video1/
│   │   │   ├── 00000.png
│   │   │   ├── 00001.png
│   │   │   └── ...
│   │   └── video2/
│   ├── train.json
│   └── test.json
└── davis/
    ├── JPEGImages_432_240/
    ├── test_masks/
    ├── train.json
    └── test.json
```

### 数据预处理

#### 1. 调整图像尺寸

将所有视频帧调整到432x240分辨率以进行训练：

```bash
python scripts/resize_videos.py \
    --input_dir datasets/youtube-vos/JPEGImages \
    --output_dir datasets/youtube-vos/JPEGImages_432_240 \
    --height 240 \
    --width 432
```

#### 2. 预计算光流（可选但推荐）

预计算光流可以显著加速训练过程：

```bash
python scripts/compute_flow.py \
    --root_path datasets/youtube-vos/JPEGImages_432_240 \
    --save_path datasets/youtube-vos/flows_432_240 \
    --height 240 \
    --width 432
```

## 训练配置

### 光流补全网络配置

配置文件：`configs/train_flowcomp.json`

```json
{
    "name": "FlowCompletion_Train",
    "model": "flowcomp",
    "gpu_ids": [0],
    "datasets": {
        "train": {
            "name": "FlowCompTrainDataset",
            "data_root": "datasets/youtube-vos",
            "flow_root": "datasets/youtube-vos/flows_432_240",
            "num_frames": 5,
            "size": [432, 240],
            "batch_size": 8
        }
    },
    "train": {
        "lr": 1e-4,
        "lr_scheme": "MultiStepLR",
        "lr_steps": [100000, 200000],
        "lr_gamma": 0.5,
        "total_iter": 300000,
        "warmup": 1000,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "save_freq": 10000,
        "val_freq": 5000
    },
    "loss": {
        "l1_weight": 1.0,
        "flow_weight": 1.0,
        "warp_weight": 0.1
    }
}
```

### ProPainter主网络配置

配置文件：`configs/train_propainter.json` (基于实际项目配置)

```json
{
    "name": "ProPainter_Train",
    "model": "propainter",
    "gpu_ids": [0],
    "strict_load": false,
    
    "datasets": {
        "train": {
            "name": "InpaintingTrain",
            "data_root": "datasets/youtube-vos/JPEGImages_432_240",
            "flow_root": "datasets/youtube-vos/flows_432_240", 
            "mask_root": "datasets/youtube-vos/test_masks",
            "num_local_frames": 5,
            "num_ref_frames": -1,
            "size": [432, 240],
            "random_reverse_clip": true,
            "flip": true
        }
    },
    
    "train": {
        "lr": 1e-4,
        "lr_scheme": "MultiStepLR",
        "lr_steps": [400000],
        "lr_gamma": 0.1,
        "total_iter": 700000,
        "warmup": -1,
        "beta1": 0.9,
        "beta2": 0.999,
        "weight_decay": 0,
        "save_freq": 10000,
        "val_freq": 5000,
        "log_freq": 100,
        "batch_size": 8
    },
    
    "loss": {
        "l1_weight": 1.0,
        "perceptual_weight": 0.1,
        "style_weight": 120.0,
        "adversarial_weight": 0.01
    }
}
```

> **注意**: 以上配置基于项目实际的配置文件，与论文中可能略有差异。

## 训练流程

### 第一步：训练光流补全网络

```bash
python train.py -c configs/train_flowcomp.json
```

训练参数说明（基于实际配置文件）：
- **学习率**: 5e-5，使用MultiStepLR调度
- **批次大小**: 8
- **训练迭代**: 700,000次
- **保存频率**: 每5,000次迭代保存一次
- **学习率衰减**: 在300k, 400k, 500k, 600k步时衰减

### 第二步：训练ProPainter主网络

```bash
python train.py -c configs/train_propainter.json
```

**重要**: 确保已经训练好光流补全网络，因为ProPainter依赖光流补全的结果。

#### 训练参数说明（基于实际配置文件）：

| 参数 | 值 | 说明 |
|------|-----|------|
| **学习率** | 1e-4 | 初始学习率 |
| **学习率调度** | MultiStepLR | 在400k步时衰减0.1倍 |
| **批次大小** | 8 | 每个GPU的批次大小 |
| **训练迭代** | 700,000次 | 总训练步数 |
| **保存频率** | 10,000步 | checkpoint保存间隔 |
| **验证频率** | 5,000步 | 验证评估间隔 |
| **日志频率** | 100步 | 训练日志输出间隔 |

#### 数据增强策略：
- **随机翻转**: 水平翻转增强
- **随机反向**: 时序反向播放
- **帧采样**: 动态帧数采样

### 恢复训练

如果训练中断，可以使用以下命令恢复：

```bash
python train.py -c configs/train_propainter.json --resume path/to/checkpoint.pth
```

## 损失函数

### 光流补全网络损失

```python
total_loss = l1_loss + flow_loss + warp_loss
```

- **L1损失**: 重建图像与真实图像的L1距离
- **光流损失**: 预测光流与真实光流的损失
- **扭曲损失**: 光流扭曲一致性损失

### ProPainter主网络损失

```python
total_loss = l1_loss + perceptual_loss + style_loss + adversarial_loss + flow_loss + warp_loss
```

- **L1损失**: 像素级重建损失
- **感知损失**: VGG特征空间的感知损失
- **风格损失**: 风格一致性损失
- **对抗损失**: GAN判别器损失
- **光流损失**: 光流一致性损失
- **扭曲损失**: 时序一致性损失

## 多GPU训练

### 数据并行

```json
{
    "gpu_ids": [0, 1, 2, 3],
    "datasets": {
        "train": {
            "batch_size": 16  // 总批次大小，会自动分配到各GPU
        }
    }
}
```

### 模型并行

对于大型模型，可以启用模型并行：

```python
# 在配置文件中设置
"model_parallel": true
```

## 训练监控

### TensorBoard日志

训练过程中会自动生成TensorBoard日志：

```bash
tensorboard --logdir experiments/
```

### 关键指标监控

- **训练损失**: 各项损失函数的变化
- **验证PSNR**: 峰值信噪比
- **验证SSIM**: 结构相似性指数
- **LPIPS**: 感知图像补丁相似性

## 训练技巧

### 1. 渐进式训练

```json
{
    "progressive": {
        "start_size": [216, 120],
        "end_size": [432, 240],
        "grow_steps": [50000, 100000, 150000]
    }
}
```

### 2. 混合精度训练

```json
{
    "fp16": true,
    "opt_level": "O1"
}
```

### 3. 学习率预热

```json
{
    "warmup": 1000,
    "warmup_lr": 1e-6
}
```

## 验证和测试

### 训练期间验证

```bash
# 验证会在训练过程中自动进行
# 频率由配置文件中的val_freq控制
```

### 独立测试

```bash
python scripts/evaluate_propainter.py \
    --dataset davis \
    --video_root datasets/davis/JPEGImages_432_240 \
    --mask_root datasets/davis/test_masks \
    --save_results
```

## 常见问题

### Q: 训练显存不足怎么办？
A: 
- 减小批次大小
- 使用梯度累积
- 启用混合精度训练
- 使用更小的输入尺寸

### Q: 训练速度太慢怎么办？
A:
- 预计算光流
- 使用多GPU训练
- 启用混合精度训练
- 优化数据加载

### Q: 损失不收敛怎么办？
A:
- 检查数据质量
- 调整学习率
- 调整损失函数权重
- 检查梯度裁剪设置

## 自定义训练

### 添加新的损失函数

```python
# 在core/loss.py中添加新损失
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        return custom_loss_calculation(pred, target)
```

### 修改网络架构

```python
# 在model/modules/中修改网络结构
class CustomProPainter(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义网络结构
```

## 训练结果分析

### 模型诊断

```bash
python scripts/analyze_model.py \
    --config configs/train_propainter.json \
    --checkpoint experiments/latest.pth
```

### 结果可视化

```bash
python scripts/visualize_results.py \
    --input_dir datasets/davis/test \
    --output_dir results/visualization