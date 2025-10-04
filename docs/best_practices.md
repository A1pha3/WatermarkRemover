# 最佳实践指南

本文档总结了使用ProPainter的最佳实践经验，帮助您获得最佳的修复效果和性能表现。

## 输入准备最佳实践

### 视频质量要求

#### 推荐的视频规格
- **分辨率**: 720p-1080p (过低影响效果，过高消耗资源)
- **帧率**: 24-30 FPS (过高的帧率意义不大)
- **编码**: H.264/H.265 (兼容性和压缩效率最佳)
- **码率**: 5-15 Mbps (保证足够的细节信息)
- **时长**: 建议单次处理不超过10分钟

#### 避免的视频特征
- **过度压缩**: 码率过低导致细节丢失
- **严重抖动**: 会影响时序一致性
- **频繁切镜**: 场景变化过快
- **极端光照**: 过曝或过暗的场景

### 掩码制作技巧

#### 高质量掩码的特征
```
✅ 边缘清晰锋利
✅ 完全覆盖目标对象
✅ 适当的膨胀边界 (2-5像素)
✅ 避免过小的孤立区域
✅ 保持时序连贯性
```

#### 制作工具推荐

**专业级工具**:
- **Adobe After Effects**: 专业视频掩码制作
- **DaVinci Resolve**: 免费专业级工具
- **Blender**: 开源3D软件，强大的遮罩功能

**入门级工具**:
- **Photoshop**: 逐帧手动制作
- **GIMP**: 免费替代方案
- **在线工具**: Remove.bg, Canva等

#### 掩码制作流程

1. **关键帧标注**: 在运动变化较大的帧上精确标注
2. **中间帧插值**: 使用工具自动插值生成中间帧
3. **手动修正**: 检查并修正插值错误
4. **边缘处理**: 适当模糊边缘避免硬边界
5. **质量检查**: 播放检查掩码的时序连续性

## 参数调优策略

### 基于场景类型的参数设置

#### 静态背景场景
```bash
# 适用于背景相对静止的场景
python inference_propainter.py \
    --video static_scene.mp4 \
    --mask mask.png \
    --neighbor_length 12 \
    --ref_stride 8 \
    --subvideo_length 80
```

#### 动态背景场景  
```bash
# 适用于背景运动较多的场景
python inference_propainter.py \
    --video dynamic_scene.mp4 \
    --mask mask.png \
    --neighbor_length 8 \
    --ref_stride 15 \
    --subvideo_length 60
```

#### 快速运动场景
```bash
# 适用于物体快速运动的场景
python inference_propainter.py \
    --video fast_motion.mp4 \
    --mask mask.png \
    --neighbor_length 6 \
    --ref_stride 20 \
    --subvideo_length 40
```

#### 复杂纹理场景
```bash
# 适用于纹理丰富的场景
python inference_propainter.py \
    --video textured_scene.mp4 \
    --mask mask.png \
    --neighbor_length 15 \
    --ref_stride 5 \
    --subvideo_length 60 \
    --resize_ratio 1.0
```

### 硬件配置优化

#### 4-6GB显存配置
```bash
# 极限内存优化
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 360 \
    --width 640 \
    --subvideo_length 20 \
    --neighbor_length 3 \
    --ref_stride 30
```

#### 8-12GB显存配置
```bash
# 平衡配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --fp16 \
    --height 480 \
    --width 720 \
    --subvideo_length 50 \
    --neighbor_length 8 \
    --ref_stride 15
```

#### 16GB+显存配置
```bash
# 高质量配置
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --height 720 \
    --width 1280 \
    --subvideo_length 80 \
    --neighbor_length 12 \
    --ref_stride 10
```

## 质量优化技巧

### 提升修复质量

#### 预处理优化
```bash
# 使用FFmpeg预处理视频
ffmpeg -i input.mp4 -vf "scale=1280:720,fps=30" -c:v libx264 -crf 18 processed.mp4
```

#### 多阶段处理
```python
# 对于复杂场景，可以分阶段处理
# 第一阶段：粗略修复
python inference_propainter.py \
    --video input.mp4 \
    --mask mask.png \
    --output stage1_result \
    --neighbor_length 8 \
    --ref_stride 15

# 第二阶段：精细化处理
python inference_propainter.py \
    --video stage1_result/inpainted_video.mp4 \
    --mask refined_mask.png \
    --output final_result \
    --neighbor_length 15 \
    --ref_stride 8
```

#### 后处理增强
```bash
# 使用FFmpeg进行后处理
ffmpeg -i result.mp4 -vf "unsharp=5:5:1.0:5:5:0.0" -c:v libx264 enhanced.mp4
```

### 时序一致性优化

#### 关键参数调整
- **增加neighbor_length**: 更好的局部时序建模
- **减少ref_stride**: 更密集的全局参考
- **适当的subvideo_length**: 平衡内存和一致性

#### 特殊场景处理
```bash
# 对于有周期性运动的场景
python inference_propainter.py \
    --video periodic_motion.mp4 \
    --mask mask.png \
    --neighbor_length 16 \
    --ref_stride 12  # 匹配运动周期
```

## 性能优化策略

### 批量处理工作流

#### 自动化批处理脚本
```bash
#!/bin/bash
# batch_process.sh

INPUT_DIR="input_videos"
MASK_DIR="masks"
OUTPUT_DIR="results"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 并行处理
for video in "$INPUT_DIR"/*.mp4; do
    basename=$(basename "$video" .mp4)
    mask="$MASK_DIR/${basename}_mask.png"
    
    if [ -f "$mask" ]; then
        echo "处理: $basename"
        python inference_propainter.py \
            --video "$video" \
            --mask "$mask" \
            --output "$OUTPUT_DIR/$basename" \
            --fp16 \
            --subvideo_length 60 &
        
        # 控制并发数
        if (( $(jobs -r | wc -l) >= 2 )); then
            wait -n
        fi
    fi
done

wait  # 等待所有任务完成
echo "批量处理完成！"
```

#### GPU资源管理
```python
# 多GPU处理策略
import os
import subprocess
from multiprocessing import Process

def process_on_gpu(gpu_id, video_list):
    """在指定GPU上处理视频列表"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    for video_path, mask_path in video_list:
        cmd = [
            'python', 'inference_propainter.py',
            '--video', video_path,
            '--mask', mask_path,
            '--fp16'
        ]
        subprocess.run(cmd)

# 使用多GPU并行处理
if __name__ == '__main__':
    video_tasks = [
        ('video1.mp4', 'mask1.png'),
        ('video2.mp4', 'mask2.png'),
        # ... more tasks
    ]
    
    # 分配任务到不同GPU
    processes = []
    for gpu_id in range(2):  # 假设有2个GPU
        gpu_tasks = video_tasks[gpu_id::2]  # 交替分配
        p = Process(target=process_on_gpu, args=(gpu_id, gpu_tasks))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
```

### 云端部署优化

#### Docker优化配置
```dockerfile
# 多阶段构建优化
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime as base

# 系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Python依赖
FROM base as python-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用层
FROM python-deps as app
WORKDIR /app
COPY . .

# 预下载模型
RUN python -c "
from utils.download_util import load_file_from_url
load_file_from_url('https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth', 'weights/')
"

# 优化启动
EXPOSE 7860
CMD ["python", "-m", "gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "300", "app:app"]
```

## 常见问题解决方案

### 质量问题诊断

#### 修复区域出现闪烁
**原因**: 时序一致性不足
**解决方案**:
```bash
# 增加邻居帧数量
--neighbor_length 15

# 减少参考帧步长  
--ref_stride 5

# 增加子视频重叠
--subvideo_length 60
```

#### 修复区域边缘明显
**原因**: 掩码边缘过硬或参数不当
**解决方案**:
1. 软化掩码边缘
2. 调整参数:
```bash
--neighbor_length 12  # 增加局部上下文
```

#### 修复区域纹理不自然
**原因**: 参考信息不足或分辨率过低
**解决方案**:
```bash
# 保持原始分辨率
--resize_ratio 1.0

# 增加参考帧密度
--ref_stride 8

# 使用更大的邻居窗口
--neighbor_length 15
```

### 性能问题解决

#### 处理速度慢
**诊断步骤**:
1. 检查GPU利用率: `nvidia-smi`
2. 检查CPU负载: `htop`
3. 检查磁盘I/O: `iotop`

**优化方案**:
```bash
# 降低处理复杂度
--neighbor_length 5
--ref_stride 20
--subvideo_length 40

# 启用加速选项
--fp16
```

#### 内存不足
**解决策略**:
```bash
# 逐步降低内存使用
1. --fp16
2. --subvideo_length 30
3. --height 360 --width 640
4. --neighbor_length 3
```

## 高级技巧

### 自定义掩码策略

#### 渐变掩码
```python
# 创建渐变边缘掩码
import cv2
import numpy as np

def create_gradient_mask(mask, gradient_width=10):
    """创建渐变边缘掩码"""
    # 距离变换
    dist = cv2.distanceTransform(255-mask, cv2.DIST_L2, 5)
    
    # 创建渐变
    gradient_mask = np.clip(dist / gradient_width * 255, 0, 255).astype(np.uint8)
    
    # 合并原掩码
    result = np.maximum(mask, gradient_mask)
    
    return result
```

#### 动态掩码调整
```python
def adjust_mask_by_content(video_path, mask_path):
    """根据视频内容动态调整掩码"""
    # 分析视频内容
    # 调整掩码大小和形状
    # 返回优化后的掩码序列
    pass
```

### 结果后处理

#### 时域平滑
```python
# 对结果进行时域平滑处理
def temporal_smooth(video_frames, window_size=3):
    """时域平滑处理"""
    smoothed = []
    for i, frame in enumerate(video_frames):
        start = max(0, i - window_size // 2)
        end = min(len(video_frames), i + window_size // 2 + 1)
        
        window_frames = video_frames[start:end]
        averaged = np.mean(window_frames, axis=0).astype(np.uint8)
        smoothed.append(averaged)
    
    return smoothed
```

#### 细节增强
```bash
# 使用FFmpeg进行细节增强
ffmpeg -i result.mp4 -vf "unsharp=5:5:0.8:3:3:0.4" enhanced.mp4
```

## 项目管理建议

### 文件组织结构
```
project_name/
├── input/
│   ├── videos/
│   └── masks/
├── output/
│   ├── results/
│   ├── intermediate/
│   └── logs/
├── configs/
│   └── custom_configs.json
└── scripts/
    ├── batch_process.sh
    └── quality_check.py
```

### 版本控制
```bash
# 使用Git管理项目配置和脚本
git init
git add configs/ scripts/ README.md
git commit -m "Initial project setup"

# 使用Git LFS管理大文件
git lfs track "*.mp4" "*.avi"
git add .gitattributes
```

### 实验记录
```python
# 实验记录模板
experiment_log = {
    "date": "2024-10-04",
    "video": "test_video.mp4",
    "mask": "test_mask.png", 
    "parameters": {
        "neighbor_length": 10,
        "ref_stride": 15,
        "subvideo_length": 60,
        "fp16": True
    },
    "results": {
        "processing_time": "5.2 minutes",
        "quality_score": 8.5,
        "issues": "slight flickering in frame 120-130"
    },
    "notes": "Good overall quality, may need to adjust neighbor_length"
}
```

---

🎯 **总结**: 遵循这些最佳实践可以显著提升ProPainter的使用效果和效率。建议根据具体场景和硬件条件选择合适的策略组合。

💡 **提示**: 建议先在小段测试视频上验证参数设置，然后再应用到完整视频处理中。