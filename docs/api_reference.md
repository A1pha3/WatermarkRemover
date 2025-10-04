# API参考文档

本文档详细描述了ProPainter的所有API接口和参数配置。

## 推理接口

### inference_propainter.py

主要的推理脚本，用于视频修复任务。

#### 基本语法

```bash
python inference_propainter.py [OPTIONS]
```

#### 必需参数

| 参数 | 类型 | 描述 |
|------|------|------|
| `--video` | str | 输入视频路径或包含帧序列的文件夹路径 |
| `--mask` | str | 掩码文件路径或包含掩码序列的文件夹路径 |

#### 可选参数

##### 输出控制
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--output` | str | `results` | 输出结果的文件夹路径 |
| `--save_frames` | bool | False | 是否保存逐帧结果 |
| `--save_flow` | bool | False | 是否保存光流结果 |

##### 视频处理
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--height` | int | None | 处理视频的目标高度 |
| `--width` | int | None | 处理视频的目标宽度 |
| `--resize_ratio` | float | 1.0 | 视频缩放比例 |
| `--fps` | float | None | 输出视频帧率 |

##### 内存优化
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--fp16` | bool | False | 使用半精度浮点数推理 |
| `--neighbor_length` | int | 10 | 局部邻居帧数量 |
| `--ref_stride` | int | 10 | 全局参考帧步长 |
| `--subvideo_length` | int | 80 | 子视频分段长度 |

##### 模型配置
| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model` | str | `ProPainter` | 使用的模型名称 |
| `--ckpt` | str | `weights/ProPainter.pth` | 模型权重文件路径 |
| `--flow_ckpt` | str | `weights/recurrent_flow_completion.pth` | 光流补全模型路径 |
| `--raft_ckpt` | str | `weights/raft-things.pth` | RAFT模型路径 |

#### 使用示例

```bash
# 基本用法
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png

# 内存优化
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --neighbor_length 5 \
    --subvideo_length 50

# 自定义输出
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --output custom_results \
    --save_frames \
    --height 480 \
    --width 640

# 高质量处理
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --neighbor_length 15 \
    --ref_stride 5 \
    --resize_ratio 1.0
```

## 训练接口

### train.py

用于训练ProPainter模型的主要脚本。

#### 基本语法

```bash
python train.py -c CONFIG_FILE [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `-c, --config` | str | 训练配置文件路径 |
| `--resume` | str | 恢复训练的checkpoint路径 |
| `--debug` | bool | 启用调试模式 |

#### 配置文件

训练配置通过JSON文件指定，主要配置文件：
- `configs/train_propainter.json`: ProPainter主模型训练配置
- `configs/train_flowcomp.json`: 光流补全网络训练配置

##### 主要配置项

```json
{
    "name": "ProPainter_Train",
    "model": "propainter",
    "gpu_ids": [0],
    "datasets": {
        "train": {
            "name": "TrainDataset",
            "data_root": "datasets/youtube-vos",
            "num_frames": 5,
            "size": [432, 240]
        }
    },
    "train": {
        "lr": 1e-4,
        "batch_size": 4,
        "total_iter": 300000,
        "save_freq": 10000
    }
}
```

## 评估接口

### evaluate_propainter.py

用于评估模型性能的脚本。

#### 基本语法

```bash
python scripts/evaluate_propainter.py [OPTIONS]
```

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `--dataset` | str | 数据集名称 (davis, youtube-vos) |
| `--video_root` | str | 视频文件根目录 |
| `--mask_root` | str | 掩码文件根目录 |
| `--save_results` | bool | 是否保存结果 |

## 工具脚本

### compute_flow.py

预计算光流以加速训练。

```bash
python scripts/compute_flow.py \
    --root_path <dataset_root> \
    --save_path <save_flow_root> \
    --height 240 \
    --width 432
```

### evaluate_flow_completion.py

评估光流补全模型。

```bash
python scripts/evaluate_flow_completion.py \
    --dataset davis \
    --video_root datasets/davis/JPEGImages_432_240 \
    --mask_root datasets/davis/test_masks
```

## Python API

### 核心类

#### ProPainter

```python
from core.propainter import ProPainter

# 初始化模型
model = ProPainter(
    model_path='weights/ProPainter.pth',
    flow_model_path='weights/recurrent_flow_completion.pth',
    raft_model_path='weights/raft-things.pth'
)

# 推理
result = model.infer(
    video_frames,  # List[np.ndarray] 或 torch.Tensor
    masks,         # List[np.ndarray] 或 torch.Tensor
    neighbor_length=10,
    ref_stride=10
)
```

#### FlowCompletionModel

```python
from core.flow_completion import FlowCompletionModel

model = FlowCompletionModel('weights/recurrent_flow_completion.pth')
completed_flow = model.complete_flow(flow, mask)
```

### 数据处理工具

#### 视频处理

```python
from utils.video_utils import VideoProcessor

processor = VideoProcessor()

# 读取视频
frames = processor.read_video('input.mp4')

# 保存视频
processor.save_video(frames, 'output.mp4', fps=30)
```

#### 掩码处理

```python
from utils.mask_utils import MaskProcessor

processor = MaskProcessor()

# 读取掩码
mask = processor.read_mask('mask.png')

# 扩展掩码
mask_dilated = processor.dilate_mask(mask, kernel_size=5)
```

## 返回值和错误处理

### 成功返回

推理成功时，结果保存在指定的输出文件夹中，包含：
- 修复后的视频文件
- 可选的逐帧图像
- 可选的中间结果（光流等）

### 错误代码

| 错误代码 | 描述 | 解决方案 |
|----------|------|----------|
| `FileNotFoundError` | 输入文件不存在 | 检查文件路径 |
| `RuntimeError` | GPU内存不足 | 使用内存优化参数 |
| `ValueError` | 参数值错误 | 检查参数范围和类型 |

## 性能基准

### 推理速度

| 分辨率 | GPU | FP32 | FP16 |
|--------|-----|------|------|
| 432x240 | RTX 3090 | 0.5s/frame | 0.3s/frame |
| 720x480 | RTX 3090 | 1.2s/frame | 0.8s/frame |
| 1280x720 | RTX 3090 | 2.5s/frame | 1.8s/frame |

### 内存使用

详细的内存使用情况请参考[内存优化指南](memory_optimization.md)。

## 版本兼容性

| ProPainter版本 | PyTorch版本 | Python版本 |
|----------------|-------------|-------------|
| 1.0.x | 1.7.1+ | 3.8+ |
| 0.9.x | 1.6.0+ | 3.7+ |