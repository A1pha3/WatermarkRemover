# 数据集准备指南

本文档详细说明如何准备和组织ProPainter所需的数据集，包括训练和评估数据。

## 数据集概述

ProPainter支持以下官方数据集：
- **YouTube-VOS**: 主要用于训练和评估
- **DAVIS**: 主要用于评估和基准测试

## YouTube-VOS数据集

### 数据集信息
- **训练集**: 3,471个视频序列
- **验证集**: 474个视频序列  
- **测试集**: 508个视频序列
- **分辨率**: 原始分辨率不一，需要调整到432x240
- **用途**: 训练ProPainter模型

### 下载和准备

#### 1. 下载原始数据

访问[官方网站](https://competitions.codalab.org/competitions/19544)下载：
- `train_all_frames.zip` - 训练集所有帧
- `valid_all_frames.zip` - 验证集所有帧  
- `test_all_frames.zip` - 测试集所有帧

#### 2. 数据解压和组织

```bash
# 创建目录结构
mkdir -p datasets/youtube-vos
cd datasets/youtube-vos

# 解压数据
unzip train_all_frames.zip
unzip valid_all_frames.zip  
unzip test_all_frames.zip

# 重新组织目录结构
mkdir JPEGImages
mv train/JPEGImages/* JPEGImages/
mv valid/JPEGImages/* JPEGImages/
mv test/JPEGImages/* JPEGImages/
```

#### 3. 调整图像尺寸

```bash
# 将所有图像调整到432x240分辨率
python scripts/resize_images.py \
    --input_dir datasets/youtube-vos/JPEGImages \
    --output_dir datasets/youtube-vos/JPEGImages_432_240 \
    --height 240 \
    --width 432 \
    --num_workers 8
```

#### 4. 下载训练掩码

从以下链接下载预制的训练掩码：
- [Google Drive](https://drive.google.com/file/d/1dFTneS_zaJAHjglxU10gYzr1-xALgHa4/view?usp=sharing)
- [百度网盘](https://pan.baidu.com/s/1JC-UKmlQfjhVtD81196cxA?pwd=87e3) (密码: 87e3)

```bash
# 解压掩码文件
unzip youtube_vos_test_masks.zip -d datasets/youtube-vos/
```

### 最终目录结构

```
datasets/youtube-vos/
├── JPEGImages_432_240/          # 调整尺寸后的图像
│   ├── 003234b8e7/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── 0034c2bbfe/
│   └── ...
├── test_masks/                  # 测试掩码
│   ├── 003234b8e7/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── ...
├── train.json                   # 训练集分割文件
└── test.json                    # 测试集分割文件
```

## DAVIS数据集

### 数据集信息
- **视频数量**: 90个高质量视频序列
- **评估子集**: 50个视频用于基准测试
- **分辨率**: 480p
- **用途**: 主要用于评估和对比

### 下载和准备

#### 1. 下载DAVIS 2017数据集

```bash
# 下载DAVIS 2017 TrainVal数据集
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip

# 解压到指定目录
unzip DAVIS-2017-trainval-480p.zip -d datasets/davis/
```

#### 2. 调整图像尺寸

```bash
# 调整到训练分辨率432x240
python scripts/resize_images.py \
    --input_dir datasets/davis/DAVIS/JPEGImages/480p \
    --output_dir datasets/davis/JPEGImages_432_240 \
    --height 240 \
    --width 432
```

#### 3. 下载评估掩码

```bash
# 从提供的链接下载DAVIS测试掩码
# 解压到datasets/davis/test_masks/
```

### 最终目录结构

```
datasets/davis/
├── JPEGImages_432_240/          # 调整尺寸后的图像
│   ├── bear/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ...
│   ├── bmx-trees/
│   └── ...
├── test_masks/                  # 测试掩码
│   ├── bear/
│   │   ├── 00000.png
│   │   ├── 00001.png
│   │   └── ...
│   └── ...
├── train.json                   # 训练集分割
└── test.json                    # 测试集分割
```

## 自定义数据集准备

### 视频数据格式

ProPainter支持两种输入格式：

#### 1. 视频文件格式
- **支持格式**: MP4, AVI, MOV等
- **推荐编码**: H.264
- **帧率**: 24-30 FPS

#### 2. 图像序列格式
- **格式**: JPG, PNG
- **命名**: 按数字顺序命名 (00000.jpg, 00001.jpg, ...)
- **组织**: 每个视频一个文件夹

### 掩码数据格式

#### 掩码要求
- **格式**: PNG（支持透明度）
- **值域**: 0-255，255表示需要修复的区域
- **尺寸**: 与对应视频帧完全一致

#### 掩码类型

##### 1. 静态掩码
适用于固定区域的视频补全：
```
mask.png  # 单个掩码文件应用于所有帧
```

##### 2. 动态掩码
适用于对象移除等需要逐帧变化的任务：
```
masks/
├── 00000.png
├── 00001.png  
├── 00002.png
└── ...
```

### 数据预处理脚本

#### 视频转换脚本

```python
# scripts/convert_video_to_frames.py
import cv2
import os
from pathlib import Path

def convert_video_to_frames(video_path, output_dir, fps=None):
    """将视频转换为图像序列"""
    cap = cv2.VideoCapture(video_path)
    
    if fps:
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(original_fps / fps)
    else:
        frame_interval = 1
    
    frame_count = 0
    saved_count = 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_dir, f"{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"转换完成: {saved_count} 帧保存到 {output_dir}")

# 使用示例
convert_video_to_frames("input.mp4", "output_frames/", fps=30)
```

#### 掩码生成脚本

```python
# scripts/generate_masks.py
import cv2
import numpy as np

def create_rectangle_mask(height, width, x, y, w, h):
    """创建矩形掩码"""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask

def create_circle_mask(height, width, center_x, center_y, radius):
    """创建圆形掩码"""
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    return mask

# 使用示例
mask = create_rectangle_mask(240, 432, 100, 50, 150, 100)
cv2.imwrite("rectangle_mask.png", mask)
```

## 数据质量检查

### 自动化质量检查脚本

```python
# scripts/check_data_quality.py
import os
import cv2
from pathlib import Path

def check_dataset_quality(dataset_root):
    """检查数据集质量"""
    issues = []
    
    # 检查图像完整性
    for video_dir in Path(dataset_root).iterdir():
        if not video_dir.is_dir():
            continue
            
        frames = sorted(video_dir.glob("*.jpg"))
        
        # 检查帧序列连续性
        expected_frames = len(frames)
        actual_frames = len([f for f in frames if f.stem.isdigit()])
        
        if expected_frames != actual_frames:
            issues.append(f"{video_dir.name}: 帧序列不连续")
        
        # 检查图像尺寸一致性
        sizes = []
        for frame_path in frames[:10]:  # 检查前10帧
            img = cv2.imread(str(frame_path))
            if img is not None:
                sizes.append(img.shape[:2])
        
        if len(set(sizes)) > 1:
            issues.append(f"{video_dir.name}: 图像尺寸不一致")
    
    return issues

# 运行检查
issues = check_dataset_quality("datasets/youtube-vos/JPEGImages_432_240")
for issue in issues:
    print(f"警告: {issue}")
```

## 数据增强

### 训练时数据增强

```python
# 在训练配置中启用数据增强
{
    "datasets": {
        "train": {
            "augmentation": {
                "horizontal_flip": 0.5,
                "color_jitter": {
                    "brightness": 0.1,
                    "contrast": 0.1,
                    "saturation": 0.1,
                    "hue": 0.05
                },
                "random_crop": {
                    "size": [432, 240],
                    "scale": [0.8, 1.0]
                }
            }
        }
    }
}
```

### 掩码增强

```python
def augment_mask(mask, dilation_range=(0, 5)):
    """增强掩码：随机膨胀"""
    kernel_size = np.random.randint(*dilation_range)
    if kernel_size > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
```

## 数据集统计信息

### 生成统计报告

```python
# scripts/dataset_statistics.py
def generate_stats(dataset_root):
    """生成数据集统计信息"""
    stats = {
        "总视频数": 0,
        "总帧数": 0,
        "平均视频长度": 0,
        "分辨率分布": {},
        "掩码覆盖率": []
    }
    
    # 统计逻辑...
    
    return stats

# 生成报告
stats = generate_stats("datasets/youtube-vos/")
print(f"数据集统计信息: {stats}")
```

## 存储优化

### 压缩存储

```bash
# 使用JPEG压缩减少存储空间
python scripts/compress_images.py \
    --input_dir datasets/youtube-vos/JPEGImages_432_240 \
    --quality 95 \
    --output_dir datasets/youtube-vos/JPEGImages_432_240_compressed
```

### 分布式存储

```bash
# 将大型数据集分割存储
python scripts/split_dataset.py \
    --input_dir datasets/youtube-vos \
    --output_dir datasets/youtube-vos-split \
    --split_size 1000  # 每个分片1000个视频
```

## 常见问题解决

### Q: 图像尺寸调整后质量下降？
A: 使用高质量的插值方法：
```python
cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
```

### Q: 掩码与视频不匹配？
A: 检查文件命名和数量：
```bash
# 检查帧数匹配
find video_dir -name "*.jpg" | wc -l
find mask_dir -name "*.png" | wc -l
```

### Q: 数据加载速度慢？
A: 优化措施：
- 使用SSD存储
- 预加载到内存
- 使用多进程数据加载
- 压缩图像质量

### Q: 内存不足？
A: 解决方案：
- 分批处理数据
- 使用图像压缩
- 实时解码而非预加载

## 数据集维护

### 定期检查

```bash
# 每月运行数据完整性检查
python scripts/check_data_integrity.py \
    --dataset_root datasets/ \
    --output_report data_check_report.json
```

### 备份策略

```bash
# 创建数据集备份
rsync -av --progress datasets/ backup/datasets/
```

### 版本控制

```bash
# 使用Git LFS管理大文件
git lfs track "*.jpg"
git lfs track "*.png"
git add .gitattributes
```

这完成了数据集准备指南。接下来让我继续创建其他重要文档。