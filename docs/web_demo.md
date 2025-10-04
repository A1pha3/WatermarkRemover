# Web演示指南

本文档详细介绍如何部署和使用ProPainter的Web演示界面，包括在线演示和本地部署。

## 演示概述

ProPainter提供多种演示方式：
- **在线演示**: Hugging Face Spaces和OpenXLab平台
- **本地演示**: 基于Gradio的交互式界面
- **API服务**: RESTful API接口

## 在线演示

### Hugging Face Spaces

访问地址: [https://huggingface.co/spaces/sczhou/ProPainter](https://huggingface.co/spaces/sczhou/ProPainter)

**功能特性**:
- 交互式对象选择和移除
- 实时预览和处理
- 支持多种视频格式
- 无需本地安装

**使用步骤**:
1. 上传待处理的视频文件
2. 使用交互式工具选择要移除的对象
3. 调整处理参数
4. 点击处理并下载结果

### OpenXLab平台

访问地址: [https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter](https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter)

**特点**:
- 中文界面支持
- 高性能GPU加速
- 批量处理功能

## 本地Web演示部署

### 环境准备

```bash
# 确保已安装ProPainter基础环境
cd ProPainter
conda activate propainter

# 安装Web演示依赖
pip install gradio==3.45.0
pip install opencv-python
pip install pillow
```

### 启动本地演示

```bash
# 进入演示目录
cd web-demos/hugging_face

# 启动Gradio应用
python app.py
```

默认访问地址: `http://localhost:7860`

### 配置参数

编辑`web-demos/hugging_face/app.py`中的配置：

```python
# 服务器配置
SERVER_CONFIG = {
    "server_name": "0.0.0.0",  # 允许外部访问
    "server_port": 7860,       # 端口号
    "debug": False,            # 调试模式
    "share": False,            # 是否生成公网链接
}

# 模型配置
MODEL_CONFIG = {
    "propainter_ckpt": "weights/ProPainter.pth",
    "flow_ckpt": "weights/recurrent_flow_completion.pth",
    "raft_ckpt": "weights/raft-things.pth",
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# 处理配置
PROCESSING_CONFIG = {
    "max_video_length": 300,   # 最大视频长度(秒)
    "max_resolution": 1280,    # 最大分辨率
    "default_fps": 30,         # 默认帧率
    "temp_dir": "temp/",       # 临时文件目录
}
```

## 界面功能详解

### 主界面布局

```
┌─────────────────────────────────────┐
│              ProPainter             │
├─────────────────┬───────────────────┤
│   视频上传区    │    参数设置区     │
├─────────────────┼───────────────────┤
│   掩码编辑器    │    预览区域       │
├─────────────────┼───────────────────┤
│   处理控制      │    结果下载       │
└─────────────────┴───────────────────┘
```

### 视频上传功能

支持的格式：
- **视频文件**: MP4, AVI, MOV, MKV
- **图像序列**: ZIP压缩包
- **最大文件大小**: 500MB
- **最大时长**: 5分钟

### 交互式掩码编辑

#### 自动对象选择
- **点击选择**: 点击要移除的对象
- **区域选择**: 拖拽选择矩形区域
- **智能分割**: 基于SAM的自动分割

#### 手动掩码绘制
- **画笔工具**: 自由绘制掣码
- **橡皮工具**: 擦除掩码区域
- **形状工具**: 绘制规则形状

### 参数调整面板

#### 基础参数
- **输出分辨率**: 自动/自定义
- **帧率**: 保持原始/自定义
- **质量设置**: 高/中/低

#### 高级参数
- **邻居长度**: 控制时序建模范围
- **参考步长**: 全局参考帧间隔
- **子视频长度**: 内存优化参数

## 使用指南

### 操作步骤

1. **上传视频**: 点击上传按钮选择视频文件
2. **获取视频信息**: 点击"Get video info"按钮
3. **选择掩码区域**: 在左侧图像上点击要移除的区域
4. **添加掩码**: 点击"Add mask"确认选择
5. **跟踪掩码**: 点击"Tracking"按钮进行全视频跟踪
6. **设置参数**: 在"ProPainter Parameters"下拉菜单中调整参数
7. **开始修复**: 点击"Inpainting"按钮开始处理
8. **下载结果**: 处理完成后下载修复后的视频

### 高级功能

#### 多掩码处理
- 可以添加多个掩码区域
- 每个掩码可以设置不同的跟踪时间范围
- 支持掩码的添加、删除和清除操作

#### 时间范围控制
- 拖拽"Track start frame"设置开始帧
- 拖拽"Track end frame"设置结束帧
- 精确控制每个掩码的作用范围

## API服务部署

### Flask API服务

创建`api_server.py`：

```python
from flask import Flask, request, jsonify, send_file
import os
import tempfile
from core.propainter import ProPainter

app = Flask(__name__)
model = ProPainter()

@app.route('/api/inpaint', methods=['POST'])
def inpaint_video():
    """视频修复API接口"""
    try:
        video_file = request.files['video']
        mask_file = request.files['mask']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = os.path.join(temp_dir, 'input.mp4')
            mask_path = os.path.join(temp_dir, 'mask.png')
            output_path = os.path.join(temp_dir, 'output.mp4')
            
            video_file.save(video_path)
            mask_file.save(mask_path)
            
            result = model.infer(
                video_path=video_path,
                mask_path=mask_path,
                output_path=output_path,
                **request.json.get('params', {})
            )
            
            if result['success']:
                return send_file(output_path, as_attachment=True)
            else:
                return jsonify({'error': result['message']}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Docker部署

创建`Dockerfile`：

```dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/download_models.py

EXPOSE 7860
CMD ["python", "web-demos/hugging_face/app.py"]
```

## 故障排除

### 常见问题

#### Q: 界面无法访问？
A: 检查防火墙设置和端口占用情况

#### Q: 上传视频失败？
A: 确认视频格式和文件大小符合要求

#### Q: 处理速度很慢？
A: 尝试降低视频分辨率或启用GPU加速

#### Q: 内存不足错误？
A: 参考[内存优化指南](memory_optimization.md)调整参数

## 定制化开发

### 修改界面样式

编辑CSS样式：

```python
# 在app.py中添加自定义CSS
css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.upload-area {
    border: 2px dashed #ccc;
    border-radius: 10px;
}
"""

with gr.Blocks(css=css) as demo:
    # 界面组件定义
    pass
```

### 添加新功能

```python
# 添加新的处理选项
def custom_processing_option():
    return gr.Checkbox(label="启用自定义处理")

# 在界面中集成
with gr.Row():
    custom_option = custom_processing_option()
```

这完成了Web演示指南的主要内容。现在让我继续创建剩余的文档。