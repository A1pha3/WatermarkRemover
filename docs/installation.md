# 安装指南

本文档提供了ProPainter的详细安装说明。

## 系统要求

### 硬件要求

#### 最低配置
- **GPU**: NVIDIA GTX 1060 6GB 或同等性能
- **系统内存**: 16GB RAM
- **存储空间**: 5GB可用空间

#### 推荐配置  
- **GPU**: NVIDIA RTX 3080 或更高（12GB+ 显存）
- **系统内存**: 32GB RAM
- **存储空间**: 10GB可用空间（包含数据集）

#### 专业配置
- **GPU**: NVIDIA RTX 4090 或 A100（24GB+ 显存）
- **系统内存**: 64GB RAM
- **存储空间**: 50GB+ SSD存储

### 软件要求

#### 必需软件
- **操作系统**: Linux (Ubuntu 18.04+), macOS (10.15+), Windows 10+
- **Python**: 3.8 - 3.10 (推荐 3.8)
- **CUDA**: 10.2+ (推荐 11.3+)
- **Git**: 用于克隆仓库

#### Python依赖
- **PyTorch**: 1.7.1+ (推荐 1.12.1)
- **Torchvision**: 0.8.2+ (推荐 0.13.1)
- **OpenCV**: 4.5.0+
- **其他依赖**: 详见 `requirements.txt`

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter
```

### 2. 创建虚拟环境

#### 使用Conda（推荐）
```bash
# 创建新的anaconda环境
conda create -n propainter python=3.8 -y
conda activate propainter
```

#### 使用venv
```bash
python3 -m venv propainter_env
source propainter_env/bin/activate  # Linux/macOS
# 或
propainter_env\Scripts\activate  # Windows
```

### 3. 安装Python依赖

```bash
pip3 install -r requirements.txt
```

#### 主要依赖包说明
- `torch>=1.7.1` - PyTorch深度学习框架
- `torchvision>=0.8.2` - 计算机视觉工具包
- `opencv-python` - 图像和视频处理
- `numpy` - 数值计算
- `scipy` - 科学计算
- `einops` - 张量操作
- `timm` - 预训练模型库
- `imageio-ffmpeg` - 视频编解码

### 4. 验证安装

运行以下命令验证安装是否成功：

```bash
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
```

## 预训练模型下载

### 自动下载（推荐）
首次运行推理时，模型会自动下载到`weights`文件夹。

### 手动下载
从[Releases V0.1.0](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0)下载预训练模型到`weights`文件夹：

```
weights/
├── ProPainter.pth
├── recurrent_flow_completion.pth
├── raft-things.pth
├── i3d_rgb_imagenet.pt  # 用于VFID评估指标
└── README.md
```

## 常见问题

### CUDA相关问题
如果遇到CUDA相关错误，请确保：
1. 安装了正确版本的CUDA驱动
2. PyTorch版本与CUDA版本兼容
3. 环境变量设置正确

### 内存不足
如果遇到内存不足问题，请参考[内存优化指南](memory_optimization.md)。

### 依赖冲突
如果遇到依赖冲突，建议：
1. 使用全新的虚拟环境
2. 按照requirements.txt中的确切版本安装
3. 考虑使用Docker环境

## Docker安装（可选）

```bash
# 构建Docker镜像
docker build -t propainter .

# 运行容器
docker run --gpus all -it propainter
```

## 下一步

安装完成后，请参考[快速开始指南](quick_start.md)开始使用ProPainter。