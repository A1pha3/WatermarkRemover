# 内存优化指南

本文档详细介绍如何优化ProPainter的内存使用，避免GPU显存不足问题，并提供不同硬件配置的推荐设置。

## 内存使用概述

ProPainter的内存消耗主要来源于：
- **视频帧缓存**: 存储输入视频帧
- **特征映射**: 神经网络中间层特征
- **光流计算**: RAFT网络的光流估计
- **注意力机制**: Transformer的注意力矩阵
- **梯度缓存**: 训练时的反向传播梯度

## GPU内存需求表

### 推理内存需求

| 分辨率 | 子视频长度 | FP32内存 | FP16内存 | 推荐GPU |
|--------|------------|----------|----------|---------|
| 320x240 | 50帧 | 3GB | 2GB | GTX 1060 6GB+ |
| 320x240 | 80帧 | 4GB | 3GB | GTX 1070 8GB+ |
| 640x480 | 50帧 | 10GB | 6GB | RTX 2080 Ti |
| 640x480 | 80帧 | 12GB | 7GB | RTX 3080 |
| 720x480 | 50帧 | 11GB | 7GB | RTX 3080 |
| 720x480 | 80帧 | 13GB | 8GB | RTX 3090 |
| 1280x720 | 50帧 | 28GB | 19GB | RTX A6000 |
| 1280x720 | 80帧 | OOM | 25GB | RTX A100 |

### 训练内存需求

| 批次大小 | 分辨率 | FP32内存 | FP16内存 | 推荐GPU |
|----------|--------|----------|----------|---------|
| 1 | 432x240 | 8GB | 5GB | RTX 2080 Ti |
| 2 | 432x240 | 14GB | 9GB | RTX 3090 |
| 4 | 432x240 | 24GB | 16GB | RTX A100 |
| 8 | 432x240 | OOM | 32GB | 多卡训练 |

## 内存优化策略

### 1. 使用半精度推理 (FP16)

最有效的内存优化方法，通常可以减少40-50%的内存使用。

```bash
# 启用FP16推理
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16
```

**优势**:
- 显著减少内存使用
- 加速推理过程
- 现代GPU都支持

**注意事项**:
- 可能轻微影响精度
- 需要GPU支持半精度计算

### 2. 调整子视频长度

通过分段处理长视频来控制内存使用。

```bash
# 减少子视频长度
python inference_propainter.py \
    --video long_video.mp4 \
    --mask mask.png \
    --subvideo_length 50  # 默认80
```

**推荐设置**:
- **8GB显存**: `--subvideo_length 30`
- **12GB显存**: `--subvideo_length 50`
- **16GB显存**: `--subvideo_length 80`
- **24GB显存**: `--subvideo_length 120`

### 3. 减少邻居帧数量

降低局部时序建模的帧数。

```bash
# 减少邻居长度
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --neighbor_length 5  # 默认10
```

**影响分析**:
- `neighbor_length=5`: 内存减少30%，质量轻微下降
- `neighbor_length=15`: 内存增加50%，质量提升有限

### 4. 增加参考帧步长

减少全局参考帧数量。

```bash
# 增加参考步长
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --ref_stride 20  # 默认10
```

**权衡考虑**:
- 步长越大，内存使用越少
- 但可能影响长期时序一致性

### 5. 降低处理分辨率

通过缩放减少内存需求。

```bash
# 方法1: 直接指定分辨率
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --height 360 \
    --width 640

# 方法2: 使用缩放比例
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --resize_ratio 0.5
```

## 极限内存优化配置

### 4GB显存配置

```bash
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 240 \
    --width 320 \
    --subvideo_length 20 \
    --neighbor_length 3 \
    --ref_stride 30
```

### 6GB显存配置

```bash
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 360 \
    --width 480 \
    --subvideo_length 40 \
    --neighbor_length 5 \
    --ref_stride 20
```

### 8GB显存配置

```bash
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 480 \
    --width 640 \
    --subvideo_length 50 \
    --neighbor_length 8 \
    --ref_stride 15
```

### 12GB+显存配置（推荐）

```bash
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 720 \
    --width 1280 \
    --subvideo_length 80 \
    --neighbor_length 10 \
    --ref_stride 10
```

## 训练内存优化

### 梯度累积

当批次大小受限时使用梯度累积。

```json
{
    "train": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,  // 等效批次大小为8
        "fp16": true
    }
}
```

### 梯度检查点

牺牲计算时间换取内存空间。

```python
# 在模型配置中启用
model = ProPainter(gradient_checkpointing=True)
```

### 混合精度训练

```json
{
    "train": {
        "fp16": true,
        "opt_level": "O1",  // 或 O2 for more aggressive optimization
        "loss_scale": "dynamic"
    }
}
```

## 内存监控工具

### GPU内存监控脚本

```python
# scripts/monitor_memory.py
import torch
import psutil
import time

def monitor_memory():
    """实时监控GPU和系统内存使用"""
    while True:
        # GPU内存
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU内存使用: {gpu_memory:.2f}GB / 缓存: {gpu_cached:.2f}GB")
        
        # 系统内存
        memory = psutil.virtual_memory()
        print(f"系统内存使用: {memory.used/1024**3:.2f}GB / {memory.total/1024**3:.2f}GB")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_memory()
```

### 内存分析工具

```bash
# 使用nvidia-smi监控GPU
watch -n 1 nvidia-smi

# 使用htop监控系统资源
htop

# Python内存分析
python -m memory_profiler inference_propainter.py --video input.mp4 --mask mask.png
```

## 自动内存优化

### 自适应配置脚本

```python
# scripts/auto_optimize_memory.py
import torch
import subprocess

def get_gpu_memory():
    """获取可用GPU内存"""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory / 1024**3

def auto_configure(video_path, mask_path):
    """根据GPU内存自动配置参数"""
    gpu_memory = get_gpu_memory()
    
    if gpu_memory < 6:
        config = {
            "fp16": True,
            "height": 240,
            "width": 320,
            "subvideo_length": 20,
            "neighbor_length": 3,
            "ref_stride": 30
        }
    elif gpu_memory < 12:
        config = {
            "fp16": True,
            "height": 480,
            "width": 640,
            "subvideo_length": 50,
            "neighbor_length": 8,
            "ref_stride": 15
        }
    else:
        config = {
            "fp16": False,
            "subvideo_length": 80,
            "neighbor_length": 10,
            "ref_stride": 10
        }
    
    # 构建命令
    cmd = ["python", "inference_propainter.py", 
           "--video", video_path, "--mask", mask_path]
    
    for key, value in config.items():
        if isinstance(value, bool) and value:
            cmd.append(f"--{key}")
        elif not isinstance(value, bool):
            cmd.extend([f"--{key}", str(value)])
    
    return cmd

# 使用示例
cmd = auto_configure("input.mp4", "mask.png")
subprocess.run(cmd)
```

## 错误处理和恢复

### 常见内存错误

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**解决方案**:
1. 启用FP16: `--fp16`
2. 减少子视频长度: `--subvideo_length 30`
3. 降低分辨率: `--height 240 --width 320`

#### 系统内存不足
```
MemoryError: Unable to allocate X.XX GiB for an array
```

**解决方案**:
1. 关闭其他应用程序
2. 增加虚拟内存
3. 使用更小的批次大小

### 内存泄漏检测

```python
# scripts/detect_memory_leak.py
import torch
import gc

def detect_memory_leak():
    """检测内存泄漏"""
    torch.cuda.empty_cache()
    gc.collect()
    
    initial_memory = torch.cuda.memory_allocated()
    
    # 运行推理
    # ... your inference code ...
    
    torch.cuda.empty_cache()
    gc.collect()
    
    final_memory = torch.cuda.memory_allocated()
    
    if final_memory > initial_memory:
        print(f"检测到内存泄漏: {(final_memory-initial_memory)/1024**2:.2f}MB")
    else:
        print("未检测到内存泄漏")
```

## 性能优化建议

### 1. 预处理优化

```python
# 预先调整图像尺寸，避免运行时缩放
def preprocess_video(video_path, target_size):
    # 一次性处理完成，避免重复计算
    pass
```

### 2. 批处理优化

```bash
# 批量处理多个小视频比单独处理大视频更高效
python batch_inference.py \
    --video_dir videos/ \
    --mask_dir masks/ \
    --batch_size 4
```

### 3. 缓存优化

```python
# 启用特征缓存
model = ProPainter(enable_feature_cache=True)
```

## 硬件升级建议

### GPU升级路径

| 当前GPU | 推荐升级 | 性能提升 |
|---------|----------|----------|
| GTX 1060 6GB | RTX 3060 12GB | 2x内存，1.5x速度 |
| RTX 2080 Ti | RTX 3090 | 2.5x内存，1.3x速度 |
| RTX 3080 | RTX 4090 | 1.5x内存，1.4x速度 |

### 系统内存建议

- **最小配置**: 16GB系统内存
- **推荐配置**: 32GB系统内存  
- **专业配置**: 64GB系统内存

## 云端部署优化

### 云服务选择

| 服务商 | 实例类型 | GPU内存 | 每小时成本 |
|--------|----------|---------|------------|
| AWS | p3.2xlarge | 16GB V100 | $3.06 |
| Google Cloud | n1-standard-4 + T4 | 16GB T4 | $0.95 |
| Azure | NC6s_v3 | 16GB V100 | $3.06 |

### 成本优化策略

1. **使用抢占式实例**减少成本
2. **批量处理**最大化利用率
3. **自动缩放**根据需求调整资源

## 故障排除检查清单

### 内存不足时的检查步骤

1. **检查GPU状态**
   ```bash
   nvidia-smi
   ```

2. **清理GPU内存**
   ```python
   torch.cuda.empty_cache()
   ```

3. **检查系统内存**
   ```bash
   free -h
   ```

4. **调整参数**
   - 启用FP16
   - 减少子视频长度
   - 降低分辨率

5. **重启进程**
   - 清理内存碎片
   - 重置CUDA上下文

这完成了内存优化指南。接下来让我继续创建其他重要文档。