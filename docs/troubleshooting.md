# 故障排除指南

本文档提供ProPainter使用过程中常见问题的解决方案和故障诊断方法。

## 常见错误分类

### 安装相关问题

#### CUDA相关错误

**错误信息**: `RuntimeError: No CUDA GPUs are available`

**原因分析**:
- CUDA驱动未正确安装
- PyTorch版本与CUDA版本不匹配
- GPU不被PyTorch识别

**解决方案**:
```bash
# 检查CUDA版本
nvidia-smi

# 检查PyTorch CUDA支持
python -c "import torch; print(torch.cuda.is_available())"

# 重新安装匹配的PyTorch版本
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 依赖包冲突

**错误信息**: `ImportError: cannot import name 'xxx' from 'xxx'`

**解决方案**:
```bash
# 创建全新环境
conda create -n propainter-clean python=3.8
conda activate propainter-clean

# 严格按照requirements.txt安装
pip install -r requirements.txt
```

### 内存相关问题

#### GPU显存不足

**错误信息**: `RuntimeError: CUDA out of memory`

**解决方案优先级**:
1. 启用FP16精度
2. 减少子视频长度
3. 降低输入分辨率
4. 减少邻居帧数量

```bash
# 极限内存优化配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 240 \
    --width 320 \
    --subvideo_length 20 \
    --neighbor_length 3
```

#### 系统内存不足

**错误信息**: `MemoryError: Unable to allocate array`

**解决方案**:
```bash
# 增加虚拟内存
sudo swapon --show
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 模型加载问题

#### 模型权重下载失败

**错误信息**: `FileNotFoundError: weights/ProPainter.pth not found`

**解决方案**:
```bash
# 手动下载模型权重
mkdir -p weights
cd weights

# 下载所有预训练模型
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
wget https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
```

#### 模型版本不兼容

**错误信息**: `RuntimeError: Error(s) in loading state_dict`

**解决方案**:
```python
# 检查模型兼容性
import torch

checkpoint = torch.load('weights/ProPainter.pth', map_location='cpu')
print("模型版本:", checkpoint.get('version', 'unknown'))
print("PyTorch版本:", torch.__version__)
```

### 输入数据问题

#### 视频格式不支持

**错误信息**: `cv2.error: OpenCV(4.x.x) error`

**解决方案**:
```bash
# 转换视频格式
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# 检查视频信息
ffprobe -v quiet -print_format json -show_format -show_streams input.mp4
```

#### 掩码格式错误

**错误信息**: `ValueError: mask shape mismatch`

**解决方案**:
```python
# 检查掩码格式
import cv2
import numpy as np

mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
print(f"掩码形状: {mask.shape}")
print(f"掩码值域: {mask.min()}-{mask.max()}")

# 标准化掩码
mask = np.where(mask > 127, 255, 0).astype(np.uint8)
cv2.imwrite('mask_fixed.png', mask)
```

### 性能问题

#### 处理速度慢

**诊断步骤**:
1. 检查GPU利用率
2. 检查CPU负载
3. 检查I/O瓶颈
4. 分析内存使用

**优化方案**:
```bash
# GPU监控
nvidia-smi -l 1

# 启用性能优化
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --neighbor_length 5 \
    --ref_stride 20
```

#### 结果质量差

**可能原因**:
- 掩码质量不好
- 输入分辨率过低
- 参数设置不当

**改进方法**:
```bash
# 提高质量的参数设置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --neighbor_length 15 \
    --ref_stride 5 \
    --resize_ratio 1.0
```

## 诊断工具

### 系统信息检查脚本

```python
# scripts/check_system.py
import torch
import cv2
import numpy as np
import psutil
import platform

def check_system():
    """系统环境检查"""
    print("=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    
    print("\n=== GPU信息 ===")
    if torch.cuda.is_available():
        print(f"CUDA可用: True")
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"显存: {torch.cuda.get_device_properties(i).total_memory/1024**3:.1f}GB")
    else:
        print("CUDA可用: False")
    
    print("\n=== 内存信息 ===")
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total/1024**3:.1f}GB")
    print(f"可用内存: {memory.available/1024**3:.1f}GB")
    
    print("\n=== 软件版本 ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print(f"NumPy: {np.__version__}")

if __name__ == "__main__":
    check_system()
```

### 模型测试脚本

```python
# scripts/test_model.py
import torch
import numpy as np
from model.propainter import InpaintGenerator

def test_model_loading():
    """测试模型加载"""
    try:
        model = InpaintGenerator()
        print("✓ 模型结构创建成功")
        
        # 测试前向传播
        dummy_input = torch.randn(1, 3, 240, 432)
        dummy_mask = torch.randn(1, 1, 240, 432)
        
        with torch.no_grad():
            output = model(dummy_input, dummy_mask)
        print("✓ 模型前向传播测试通过")
        
        return True
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

if __name__ == "__main__":
    test_model_loading()
```

### 数据验证脚本

```python
# scripts/validate_data.py
import cv2
import os
import numpy as np

def validate_video_mask_pair(video_path, mask_path):
    """验证视频和掩码匹配性"""
    issues = []
    
    # 检查文件存在性
    if not os.path.exists(video_path):
        issues.append(f"视频文件不存在: {video_path}")
    if not os.path.exists(mask_path):
        issues.append(f"掩码文件不存在: {mask_path}")
    
    if issues:
        return issues
    
    # 检查视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        issues.append("无法打开视频文件")
        return issues
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # 检查掩码
    if os.path.isfile(mask_path):
        # 单一掩码文件
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            issues.append("无法读取掩码文件")
        elif mask.shape != (height, width):
            issues.append(f"掩码尺寸不匹配: 期望{(height, width)}, 实际{mask.shape}")
    else:
        # 掩码序列文件夹
        mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.png')])
        if len(mask_files) != frame_count:
            issues.append(f"掩码数量不匹配: 期望{frame_count}, 实际{len(mask_files)}")
    
    return issues

# 使用示例
issues = validate_video_mask_pair("input.mp4", "mask.png")
for issue in issues:
    print(f"警告: {issue}")
```

## 性能优化诊断

### GPU利用率分析

```bash
# 监控GPU使用情况
nvidia-smi dmon -s pucvmet -d 1

# 分析GPU利用率低的原因
# 1. CPU瓶颈 - 增加数据加载进程数
# 2. I/O瓶颈 - 使用SSD存储
# 3. 内存瓶颈 - 减少批次大小
```

### 内存使用分析

```python
# scripts/profile_memory.py
import torch
import psutil
import time
from memory_profiler import profile

@profile
def memory_intensive_operation():
    """内存密集操作分析"""
    # 模拟推理过程
    data = torch.randn(1, 3, 240, 432).cuda()
    # 处理逻辑...
    return data

# 运行内存分析
# python -m memory_profiler scripts/profile_memory.py
```

## 错误日志分析

### 日志收集脚本

```python
# scripts/collect_logs.py
import os
import sys
import traceback
import datetime

class ErrorLogger:
    def __init__(self, log_file="error.log"):
        self.log_file = log_file
    
    def log_error(self, error, context=""):
        """记录错误信息"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(self.log_file, "a") as f:
            f.write(f"\n{'='*50}\n")
            f.write(f"时间: {timestamp}\n")
            f.write(f"错误: {str(error)}\n")
            if context:
                f.write(f"上下文: {context}\n")
            f.write(f"堆栈跟踪:\n{traceback.format_exc()}\n")
    
    def log_system_info(self):
        """记录系统信息"""
        with open(self.log_file, "a") as f:
            f.write(f"Python版本: {sys.version}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n")
            # 更多系统信息...

# 使用示例
logger = ErrorLogger()
try:
    # 可能出错的代码
    pass
except Exception as e:
    logger.log_error(e, "推理过程中")
```

## 社区支持

### 获取帮助的渠道

1. **GitHub Issues**: 报告Bug和功能请求
2. **讨论区**: 技术交流和问题讨论
3. **文档**: 查阅详细的使用说明

### 报告问题的最佳实践

提交Issue时请包含：
- 详细的错误信息
- 复现步骤
- 系统环境信息
- 输入数据示例（如果可能）

```bash
# 生成诊断报告
python scripts/generate_diagnostic_report.py > diagnostic_report.txt
```

### 常用命令速查

```bash
# 快速诊断
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi
free -h

# 清理缓存
python -c "import torch; torch.cuda.empty_cache()"
pip cache purge

# 重置环境
conda deactivate
conda remove -n propainter --all
# 重新安装...
```

## 高级调试技巧

### 使用调试器

```python
# 在代码中设置断点
import pdb; pdb.set_trace()

# 或使用IPython调试器
import IPython; IPython.embed()
```

### 可视化调试

```python
# 可视化中间结果
import matplotlib.pyplot as plt

def visualize_tensor(tensor, title=""):
    """可视化张量"""
    if tensor.dim() == 4:  # BCHW
        tensor = tensor[0]  # 取第一个batch
    if tensor.dim() == 3:  # CHW
        if tensor.shape[0] == 3:  # RGB
            tensor = tensor.permute(1, 2, 0)
        else:
            tensor = tensor[0]  # 取第一个通道
    
    plt.figure(figsize=(10, 8))
    plt.imshow(tensor.cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.colorbar()
    plt.show()
```

### 性能剖析

```python
# 使用PyTorch Profiler
with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    # 运行代码
    pass
```

这完成了故障排除指南。现在让我创建最后一个文档：