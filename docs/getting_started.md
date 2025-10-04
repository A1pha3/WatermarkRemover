# 新手入门指南

欢迎使用ProPainter！本指南将帮助您快速了解和使用这个强大的视频修复工具。

## 什么是ProPainter？

ProPainter是一个基于深度学习的视频修复工具，发表于ICCV 2023。它能够智能地从视频中移除不需要的对象，或填补视频中的缺失区域，同时保持高质量的视觉效果和时序一致性。

### 🎯 核心能力

- **智能对象移除**: 自动移除视频中的人物、标志、水印等
- **视频内容补全**: 填补损坏或缺失的视频区域
- **时序一致性**: 确保修复结果在时间维度上自然流畅
- **高质量输出**: 保持原始视频的清晰度和细节

### 🏆 技术优势

- **最先进算法**: 基于改进的传播机制和Transformer架构
- **论文验证**: ICCV 2023发表，性能经过学术验证
- **易于使用**: 提供简单的命令行接口和Web界面
- **内存优化**: 支持不同硬件配置的优化策略

## 5分钟快速体验

### 在线体验（推荐新手）

无需安装，直接在浏览器中体验：

1. **Hugging Face Spaces**: [https://huggingface.co/spaces/sczhou/ProPainter](https://huggingface.co/spaces/sczhou/ProPainter)
   - 交互式界面，支持对象选择
   - 实时预览处理效果
   - 无需本地安装

2. **OpenXLab平台**: [https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter](https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter)
   - 中文界面支持
   - 高性能GPU加速

### 本地快速体验

如果您想在本地运行，请按以下步骤操作：

#### 步骤1: 环境准备
```bash
# 克隆项目
git clone https://github.com/sczhou/ProPainter.git
cd ProPainter

# 创建环境
conda create -n propainter python=3.8 -y
conda activate propainter

# 安装依赖
pip install -r requirements.txt
```

#### 步骤2: 快速测试
```bash
# 使用提供的示例数据进行测试
python inference_propainter.py \
    --video inputs/object_removal/bmx-trees \
    --mask inputs/object_removal/bmx-trees_mask
```

#### 步骤3: 查看结果
处理完成后，在 `results` 文件夹中查看修复后的视频。

## 基本概念

### 输入文件

ProPainter需要两个主要输入：

1. **视频文件**: 需要修复的原始视频
   - 支持格式: MP4, AVI, MOV等
   - 或者图像序列文件夹

2. **掩码文件**: 标记需要修复的区域
   - PNG格式，白色区域(255)表示需要修复的部分
   - 可以是单个掩码文件或逐帧掩码序列

### 处理流程

```
输入视频 + 掩码 → ProPainter处理 → 修复后的视频
```

### 输出结果

- **主要输出**: 修复后的视频文件
- **可选输出**: 逐帧图像、中间结果等

## 常见使用场景

### 1. 移除视频中的人物
```bash
python inference_propainter.py \
    --video your_video.mp4 \
    --mask person_mask.png
```

### 2. 移除水印或标志
```bash
python inference_propainter.py \
    --video branded_video.mp4 \
    --mask watermark_mask.png
```

### 3. 修复视频损坏区域
```bash
python inference_propainter.py \
    --video damaged_video.mp4 \
    --mask damage_mask.png
```

### 4. 填补视频空白区域
```bash
python inference_propainter.py \
    --video incomplete_video.mp4 \
    --mask missing_area_mask.png
```

## 制作掩码的方法

### 方法1: 使用图像编辑软件
- **Photoshop**: 使用选择工具创建精确掩码
- **GIMP**: 免费替代方案，功能完整
- **Paint.NET**: 轻量级选择，易于使用

### 方法2: 在线工具
- **Remove.bg**: 自动背景移除
- **Canva**: 在线图像编辑器
- **Photopea**: 免费的在线Photoshop替代品

### 方法3: 编程生成
```python
import cv2
import numpy as np

# 创建简单的矩形掩码
def create_rectangle_mask(width, height, x, y, w, h):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255
    return mask

mask = create_rectangle_mask(640, 480, 100, 100, 200, 150)
cv2.imwrite('mask.png', mask)
```

## 性能优化建议

### 显存不足时
```bash
# 启用半精度推理
--fp16

# 减少子视频长度
--subvideo_length 30

# 降低处理分辨率
--height 360 --width 640
```

### 提升处理速度
```bash
# 减少邻居帧数量
--neighbor_length 5

# 增加参考帧步长
--ref_stride 20
```

## 常见问题快速解答

### Q: 处理时间很长怎么办？
A: 尝试降低视频分辨率或启用 `--fp16` 参数。

### Q: 显存不足错误？
A: 参考[内存优化指南](memory_optimization.md)调整参数。

### Q: 结果质量不满意？
A: 检查掩码质量，确保边缘清晰且覆盖完整。

### Q: 支持哪些视频格式？
A: 支持MP4、AVI、MOV等常见格式，推荐使用MP4。

## 下一步学习

根据您的需求，建议继续阅读：

### 基础用户
- [安装指南](installation.md) - 详细的环境配置
- [API参考](api_reference.md) - 了解所有参数选项
- [常见问题](faq.md) - 解决常见使用问题

### 进阶用户  
- [内存优化](memory_optimization.md) - 优化显存使用
- [性能优化](performance_guide.md) - 提升处理性能
- [Web演示](web_demo.md) - 部署Web界面

### 专业用户
- [训练指南](training.md) - 训练自定义模型
- [评估指南](evaluation.md) - 评估模型性能
- [贡献指南](contributing.md) - 参与项目开发

## 获取帮助

### 官方资源
- **项目主页**: [https://shangchenzhou.com/projects/ProPainter/](https://shangchenzhou.com/projects/ProPainter/)
- **GitHub仓库**: [https://github.com/sczhou/ProPainter](https://github.com/sczhou/ProPainter)
- **学术论文**: [ICCV 2023 论文](https://arxiv.org/abs/2309.03897)

### 社区支持
- **Issue报告**: [GitHub Issues](https://github.com/sczhou/ProPainter/issues)
- **技术讨论**: [GitHub Discussions](https://github.com/sczhou/ProPainter/discussions)
- **在线演示**: 直接体验功能效果

### 联系方式
- **项目负责人**: shangchenzhou@gmail.com
- **学术合作**: 通过邮件联系讨论

---

🎉 **恭喜！** 您已经了解了ProPainter的基本使用方法。现在可以开始您的视频修复之旅了！

💡 **提示**: 建议先使用在线演示熟悉功能，然后根据需要进行本地安装和深度使用。