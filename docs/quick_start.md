# 快速开始指南

本指南将帮助您快速上手ProPainter，进行视频修复任务。

## 概述

ProPainter支持两种主要的视频修复任务：
- **对象移除**: 从视频中移除不需要的对象
- **视频补全**: 填补视频中的缺失区域

## 准备工作

确保您已经完成了[安装](installation.md)并下载了预训练模型。

## 快速测试

我们在`inputs`文件夹中提供了示例数据，您可以直接运行以下命令进行测试。

### 对象移除示例

```bash
# 移除BMX视频中的树木
python inference_propainter.py \
    --video inputs/object_removal/bmx-trees \
    --mask inputs/object_removal/bmx-trees_mask
```

### 视频补全示例

```bash
# 补全汽车视频中的方形区域
python inference_propainter.py \
    --video inputs/video_completion/running_car.mp4 \
    --mask inputs/video_completion/mask_square.png \
    --height 240 \
    --width 432
```

> **注意**: 确保输入路径正确，示例数据位于项目的 `inputs` 文件夹中。

## 处理自定义视频

### 输入格式

ProPainter支持以下输入格式：
- **视频文件**: MP4, AVI等常见格式
- **图像序列**: 按帧分割的图像文件夹

### 掩码格式

- **单一掩码**: PNG格式，用于整个视频的固定区域
- **逐帧掩码**: 每帧对应一个掩码文件

### 基本用法

```bash
python inference_propainter.py \
    --video /path/to/your/video.mp4 \
    --mask /path/to/your/mask.png \
    --output /path/to/output/folder
```

### 主要参数说明

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--video` | 输入视频路径或图像文件夹 | 必需 |
| `--mask` | 掩码文件或文件夹路径 | 必需 |
| `--output` | 输出文件夹路径 | `results` |
| `--height` | 处理视频的高度 | 原始高度 |
| `--width` | 处理视频的宽度 | 原始宽度 |
| `--fp16` | 使用半精度推理（节省显存） | False |

## 内存优化选项

如果遇到显存不足问题，可以使用以下参数：

```bash
python inference_propainter.py \
    --video your_video.mp4 \
    --mask your_mask.png \
    --height 320 \
    --width 576 \
    --fp16 \
    --neighbor_length 5 \
    --ref_stride 20 \
    --subvideo_length 50
```

### 内存优化参数

| 参数 | 描述 | 建议值 |
|------|------|--------|
| `--neighbor_length` | 局部邻居数量 | 5-10 |
| `--ref_stride` | 全局参考步长 | 10-20 |
| `--resize_ratio` | 缩放比例 | 0.5-1.0 |
| `--subvideo_length` | 子视频长度 | 50-80 |

## 交互式演示

### 在线演示
- [Hugging Face Demo](https://huggingface.co/spaces/sczhou/ProPainter)
- [OpenXLab Demo](https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter)

### 本地演示
```bash
cd web-demos/hugging_face
python app.py
```

详细说明请参考[Web演示指南](web_demo.md)。

## 输出结果

处理完成后，您将在输出文件夹中找到：
- `inpainted_video.mp4`: 修复后的视频
- `frames/`: 逐帧结果（如果启用）
- `flow/`: 光流结果（如果启用保存）

## 性能提示

1. **选择合适的分辨率**: 较低分辨率处理更快，显存需求更少
2. **使用FP16**: 在支持的GPU上可以显著减少显存使用
3. **调整子视频长度**: 长视频可以分段处理以节省显存
4. **预处理掩码**: 确保掩码质量良好，边缘清晰

## 常见问题

### Q: 处理很慢怎么办？
A: 尝试降低分辨率、使用FP16精度，或减少邻居长度。

### Q: 显存不足怎么办？
A: 参考[内存优化指南](memory_optimization.md)调整相关参数。

### Q: 结果质量不好怎么办？
A: 检查掩码质量，尝试调整参数，或参考[故障排除](troubleshooting.md)。

## 下一步

- 了解更多参数配置：[API参考](api_reference.md)
- 训练自定义模型：[训练指南](training.md)
- 评估模型性能：[评估指南](evaluation.md)