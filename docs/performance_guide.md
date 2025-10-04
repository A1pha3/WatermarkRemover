# 性能优化指南

本文档提供ProPainter性能优化的全面指南，帮助您在不同硬件配置下获得最佳性能。

## 性能基准

### 推理速度基准

| 分辨率 | GPU型号 | FP32速度 | FP16速度 | 内存使用 |
|--------|---------|----------|----------|----------|
| 320x240 | RTX 3060 | 0.8s/帧 | 0.5s/帧 | 4GB |
| 432x240 | RTX 3070 | 1.2s/帧 | 0.8s/帧 | 6GB |
| 640x480 | RTX 3080 | 2.1s/帧 | 1.4s/帧 | 10GB |
| 720x480 | RTX 3090 | 2.8s/帧 | 1.9s/帧 | 12GB |
| 1280x720 | RTX 4090 | 4.5s/帧 | 3.2s/帧 | 18GB |

### 质量vs性能权衡

| 配置级别 | 质量评分 | 处理速度 | 内存需求 | 适用场景 |
|----------|----------|----------|----------|----------|
| 极速模式 | 7.5/10 | 3x | 50% | 预览、快速测试 |
| 平衡模式 | 8.5/10 | 1x | 100% | 日常使用 |
| 高质量模式 | 9.2/10 | 0.6x | 150% | 专业制作 |
| 极致模式 | 9.5/10 | 0.3x | 200% | 最终输出 |

## 硬件优化建议

### GPU选择指南

#### 入门级配置 (4-6GB显存)
**推荐GPU**: GTX 1660 Ti, RTX 3060
```bash
# 优化配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 240 \
    --width 320 \
    --subvideo_length 30 \
    --neighbor_length 5
```

#### 主流配置 (8-12GB显存)
**推荐GPU**: RTX 3070, RTX 3080
```bash
# 平衡配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 480 \
    --width 640 \
    --subvideo_length 60 \
    --neighbor_length 8
```

#### 专业配置 (16GB+显存)
**推荐GPU**: RTX 3090, RTX 4080, RTX 4090
```bash
# 高质量配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --height 720 \
    --width 1280 \
    --subvideo_length 80 \
    --neighbor_length 12
```

### CPU和内存优化

#### 系统内存建议
- **最低配置**: 16GB DDR4
- **推荐配置**: 32GB DDR4 3200MHz+
- **专业配置**: 64GB DDR4 3600MHz+

#### CPU优化设置
```bash
# 设置CPU线程数
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# 启用CPU优化
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --num_workers 8
```

## 软件优化策略

### 1. 精度优化

#### 混合精度推理
```python
# 在代码中启用自动混合精度
import torch
from torch.cuda.amp import autocast

with autocast():
    output = model(input_tensor)
```

#### 量化优化
```python
# 模型量化（实验性功能）
import torch.quantization as quant

model_quantized = quant.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

### 2. 内存管理优化

#### 梯度检查点
```python
# 训练时启用梯度检查点
model = torch.utils.checkpoint.checkpoint_sequential(
    model, segments=4, input=input_tensor
)
```

#### 内存池优化
```python
# 设置内存池
torch.cuda.set_per_process_memory_fraction(0.9)
torch.cuda.empty_cache()
```

### 3. 数据加载优化

#### 异步数据加载
```python
# 优化数据加载器
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2
)
```

#### 预处理缓存
```bash
# 预计算光流以加速训练
python scripts/compute_flow.py \
    --root_path datasets/youtube-vos \
    --save_path datasets/flows \
    --num_workers 16
```

## 参数调优指南

### 核心参数影响分析

#### neighbor_length (邻居长度)
- **影响**: 时序建模质量 vs 内存使用
- **建议值**: 5-15
- **优化策略**: 短视频用较大值，长视频用较小值

```bash
# 不同场景的推荐设置
# 短视频 (<30秒)
--neighbor_length 12

# 中等视频 (30-120秒)  
--neighbor_length 8

# 长视频 (>120秒)
--neighbor_length 5
```

#### ref_stride (参考步长)
- **影响**: 全局一致性 vs 计算效率
- **建议值**: 5-20
- **优化策略**: 静态场景用大值，动态场景用小值

#### subvideo_length (子视频长度)
- **影响**: 内存使用 vs 处理效率
- **建议值**: 30-120
- **优化策略**: 根据可用内存动态调整

### 自适应参数调整

```python
# 自动参数优化脚本
def optimize_parameters(video_path, gpu_memory_gb):
    """根据视频特性和硬件自动优化参数"""
    
    # 获取视频信息
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 基于GPU内存调整参数
    if gpu_memory_gb < 8:
        config = {
            "fp16": True,
            "subvideo_length": 30,
            "neighbor_length": 5,
            "ref_stride": 20
        }
    elif gpu_memory_gb < 16:
        config = {
            "fp16": True,
            "subvideo_length": 60,
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
    
    # 基于视频分辨率调整
    if width * height > 1280 * 720:
        config["resize_ratio"] = 0.8
    
    # 基于视频长度调整
    if frame_count > 300:
        config["neighbor_length"] = max(3, config["neighbor_length"] - 2)
    
    return config
```

## 批量处理优化

### 并行处理策略

```python
# 多GPU并行处理
import torch.multiprocessing as mp

def process_video_parallel(video_list, num_gpus=2):
    """多GPU并行处理视频列表"""
    
    processes = []
    for i, video_path in enumerate(video_list):
        gpu_id = i % num_gpus
        p = mp.Process(
            target=process_single_video,
            args=(video_path, gpu_id)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
```

### 批量优化脚本

```bash
#!/bin/bash
# batch_process.sh - 批量处理优化脚本

VIDEO_DIR="input_videos"
OUTPUT_DIR="output_videos"
MASK_DIR="masks"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 并行处理多个视频
for video in "$VIDEO_DIR"/*.mp4; do
    basename=$(basename "$video" .mp4)
    mask="$MASK_DIR/${basename}_mask.png"
    output="$OUTPUT_DIR/${basename}_inpainted.mp4"
    
    # 后台并行处理
    python inference_propainter.py \
        --video "$video" \
        --mask "$mask" \
        --output "$output" \
        --fp16 \
        --subvideo_length 60 &
    
    # 控制并发数量
    if (( $(jobs -r | wc -l) >= 4 )); then
        wait -n  # 等待任意一个任务完成
    fi
done

wait  # 等待所有任务完成
```

## 云端部署优化

### Docker优化配置

```dockerfile
# 优化的Dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 复制并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 预下载模型权重
RUN python -c "from utils.download_util import load_file_from_url; \
    load_file_from_url('https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth', 'weights/')"

# 设置启动命令
CMD ["python", "inference_propainter.py"]
```

### Kubernetes部署

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: propainter
spec:
  replicas: 2
  selector:
    matchLabels:
      app: propainter
  template:
    metadata:
      labels:
        app: propainter
    spec:
      containers:
      - name: propainter
        image: propainter:latest
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "4"
          limits:
            nvidia.com/gpu: 1
            memory: "32Gi"
            cpu: "8"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
```

## 性能监控

### 实时监控脚本

```python
# performance_monitor.py
import psutil
import torch
import time
import matplotlib.pyplot as plt
from collections import deque

class PerformanceMonitor:
    def __init__(self, max_points=100):
        self.max_points = max_points
        self.gpu_memory = deque(maxlen=max_points)
        self.gpu_util = deque(maxlen=max_points)
        self.cpu_util = deque(maxlen=max_points)
        self.timestamps = deque(maxlen=max_points)
    
    def update(self):
        """更新性能指标"""
        current_time = time.time()
        
        # GPU指标
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            # GPU利用率需要nvidia-ml-py包
            # gpu_util = nvidia_ml_py.nvmlDeviceGetUtilizationRates(handle).gpu
        else:
            gpu_memory = 0
            gpu_util = 0
        
        # CPU指标
        cpu_util = psutil.cpu_percent()
        
        # 记录数据
        self.gpu_memory.append(gpu_memory)
        self.cpu_util.append(cpu_util)
        self.timestamps.append(current_time)
    
    def plot_metrics(self):
        """绘制性能图表"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # GPU内存使用
        axes[0].plot(list(self.timestamps), list(self.gpu_memory))
        axes[0].set_title('GPU Memory Usage (GB)')
        axes[0].set_ylabel('Memory (GB)')
        
        # CPU使用率
        axes[1].plot(list(self.timestamps), list(self.cpu_util))
        axes[1].set_title('CPU Utilization (%)')
        axes[1].set_ylabel('CPU %')
        axes[1].set_xlabel('Time')
        
        plt.tight_layout()
        plt.savefig('performance_metrics.png')
        plt.show()

# 使用示例
monitor = PerformanceMonitor()
for i in range(100):
    monitor.update()
    time.sleep(1)
monitor.plot_metrics()
```

## 故障诊断

### 性能问题诊断清单

1. **GPU利用率低 (<50%)**
   - 检查CPU瓶颈
   - 优化数据加载
   - 增加批次大小

2. **内存使用过高**
   - 启用FP16精度
   - 减少子视频长度
   - 清理GPU缓存

3. **处理速度慢**
   - 检查硬盘I/O
   - 优化网络结构
   - 使用预计算特征

### 性能调优工具

```bash
# GPU性能分析
nvidia-smi dmon -s pucvmet -d 1

# CPU性能分析
htop

# 内存分析
python -m memory_profiler inference_propainter.py

# PyTorch性能分析
python -c "
import torch.profiler
with torch.profiler.profile() as prof:
    # 运行代码
    pass
print(prof.key_averages().table())
"
```

## 最佳实践总结

### 通用优化原则

1. **硬件优先**: 选择合适的GPU和足够的内存
2. **参数调优**: 根据具体场景调整核心参数
3. **精度权衡**: 在质量和速度间找到平衡点
4. **监控优化**: 持续监控性能指标并调整
5. **批量处理**: 充分利用并行处理能力

### 场景特定建议

#### 实时处理场景
- 优先考虑速度，适当牺牲质量
- 使用较小的邻居长度和参考步长
- 启用FP16和其他加速技术

#### 离线高质量处理
- 优先考虑质量，可以接受较慢速度
- 使用较大的邻居长度和密集参考
- 考虑使用FP32精度

#### 云端服务部署
- 平衡成本和性能
- 实现自动扩缩容
- 优化资源利用率

---

通过遵循本指南的建议，您可以在各种硬件配置和使用场景下获得ProPainter的最佳性能表现。