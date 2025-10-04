# 常见问题 (FAQ)

本文档收集了ProPainter使用过程中的常见问题和解答，帮助您快速解决遇到的问题。

## 安装和环境问题

### Q1: 安装时出现CUDA错误怎么办？

**A**: 这通常是CUDA版本不匹配导致的：

1. **检查CUDA版本**:
   ```bash
   nvidia-smi  # 查看驱动支持的CUDA版本
   nvcc --version  # 查看已安装的CUDA版本
   ```

2. **安装匹配的PyTorch版本**:
   ```bash
   # CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **验证安装**:
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

### Q2: 为什么模型下载很慢或失败？

**A**: 可以手动下载模型权重：

```bash
# 创建weights目录
mkdir -p weights

# 手动下载模型文件
wget -O weights/ProPainter.pth https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth
wget -O weights/recurrent_flow_completion.pth https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth
wget -O weights/raft-things.pth https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth
```

### Q3: 在macOS上安装失败怎么办？

**A**: macOS用户需要注意：

1. **使用CPU版本**（如果没有NVIDIA GPU）:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

2. **安装额外依赖**:
   ```bash
   brew install ffmpeg
   ```

3. **使用MPS加速**（Apple Silicon）:
   ```python
   # 在代码中启用MPS
   device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
   ```

## 使用和操作问题

### Q4: 如何处理不同格式的视频？

**A**: ProPainter支持多种视频格式：

**支持的输入格式**:
- 视频文件: MP4, AVI, MOV, MKV, WMV
- 图像序列: JPG, PNG图像文件夹

**格式转换**:
```bash
# 转换为MP4格式
ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4

# 提取图像序列
ffmpeg -i input.mp4 frames/frame_%05d.jpg

# 图像序列转视频
ffmpeg -r 30 -i frames/frame_%05d.jpg -c:v libx264 output.mp4
```

### Q5: 掩码应该如何制作？

**A**: 掩码制作要点：

**格式要求**:
- 格式: PNG（支持透明度更好）
- 尺寸: 与视频帧完全一致
- 值域: 0-255，255表示需要修复的区域

**制作工具**:
1. **Photoshop/GIMP**: 手动绘制精确掩码
2. **DaVinci Resolve**: 视频编辑软件的遮罩功能
3. **OpenCV脚本**: 程序化生成掩码

**示例代码**:
```python
import cv2
import numpy as np

# 创建圆形掩码
def create_circle_mask(height, width, center, radius):
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask

# 使用
mask = create_circle_mask(480, 640, (320, 240), 50)
cv2.imwrite('circle_mask.png', mask)
```

### Q6: 处理结果质量不好怎么办？

**A**: 提升质量的方法：

1. **检查掩码质量**:
   - 确保掩码边缘清晰
   - 掩码区域不要过大（建议<30%画面）
   - 避免掩码覆盖重要结构边缘

2. **调整参数设置**:
   ```bash
   # 高质量设置
   python inference_propainter.py \
       --video input.mp4 \
       --mask mask.png \
       --neighbor_length 15 \
       --ref_stride 5 \
       --resize_ratio 1.0
   ```

3. **预处理优化**:
   - 确保输入视频质量良好
   - 避免过度压缩的视频
   - 考虑提升输入分辨率

## 性能和内存问题

### Q7: 显存不足错误如何解决？

**A**: 内存优化策略（按优先级）：

1. **启用FP16精度**:
   ```bash
   --fp16
   ```

2. **减少子视频长度**:
   ```bash
   --subvideo_length 30  # 默认80
   ```

3. **降低处理分辨率**:
   ```bash
   --height 360 --width 640  # 或使用--resize_ratio 0.5
   ```

4. **调整其他参数**:
   ```bash
   --neighbor_length 5  # 默认10
   --ref_stride 20      # 默认10
   ```

### Q8: 处理速度太慢怎么办？

**A**: 加速方法：

1. **硬件升级**: 使用更强的GPU
2. **参数优化**: 
   ```bash
   --fp16 --neighbor_length 5 --ref_stride 15
   ```
3. **分辨率调整**: 降低输入分辨率
4. **预处理**: 预计算光流（训练时）

### Q9: 如何批量处理多个视频？

**A**: 批量处理脚本：

```bash
#!/bin/bash
# 批量处理脚本
for video in input_dir/*.mp4; do
    basename=$(basename "$video" .mp4)
    python inference_propainter.py \
        --video "$video" \
        --mask "masks/${basename}_mask.png" \
        --output "results/${basename}_result.mp4"
done
```

## 训练和模型问题

### Q10: 如何训练自定义数据？

**A**: 训练步骤：

1. **准备数据集**:
   ```
   datasets/my_dataset/
   ├── JPEGImages_432_240/
   ├── test_masks/
   ├── train.json
   └── test.json
   ```

2. **修改配置文件**:
   ```json
   {
       "train_data_loader": {
           "video_root": "datasets/my_dataset/JPEGImages_432_240",
           "flow_root": "datasets/my_dataset/flows"
       }
   }
   ```

3. **开始训练**:
   ```bash
   python train.py -c configs/train_propainter.json
   ```

### Q11: 训练时显存不足怎么办？

**A**: 训练优化：

1. **减少批次大小**:
   ```json
   "batch_size": 2  // 默认8
   ```

2. **启用梯度累积**:
   ```json
   "gradient_accumulation_steps": 4
   ```

3. **使用混合精度**:
   ```json
   "fp16": true
   ```

## Web演示问题

### Q12: 本地Web演示无法启动？

**A**: 检查以下问题：

1. **安装Gradio**:
   ```bash
   pip install gradio==3.45.0
   ```

2. **检查端口占用**:
   ```bash
   lsof -i :7860  # 检查7860端口
   ```

3. **修改启动配置**:
   ```python
   # 在app.py中修改
   demo.launch(server_name="0.0.0.0", server_port=7861)
   ```

### Q13: Web界面上传视频失败？

**A**: 可能的原因：

1. **文件大小限制**: 默认限制500MB
2. **格式不支持**: 确保使用MP4格式
3. **网络问题**: 检查网络连接稳定性

## 高级使用问题

### Q14: 如何集成到其他项目？

**A**: API集成示例：

```python
# 作为模块导入
from inference_propainter import ProPainterInference

# 创建推理器
inpainter = ProPainterInference(
    propainter_ckpt='weights/ProPainter.pth',
    device='cuda'
)

# 处理视频
result = inpainter.infer(
    video_path='input.mp4',
    mask_path='mask.png',
    **kwargs
)
```

### Q15: 如何自定义网络结构？

**A**: 修改模型结构：

1. **修改配置文件**:
   ```json
   "model": {
       "net": "custom_propainter",
       "custom_param": "value"
   }
   ```

2. **实现自定义模型**:
   ```python
   # 在model/目录下创建custom_propainter.py
   class CustomProPainter(nn.Module):
       def __init__(self, **kwargs):
           super().__init__()
           # 自定义结构
   ```

## 错误代码参考

### 常见错误代码

| 错误代码 | 含义 | 解决方案 |
|----------|------|----------|
| `CUDA_ERROR_OUT_OF_MEMORY` | GPU显存不足 | 参考Q7 |
| `FileNotFoundError` | 文件路径错误 | 检查文件路径 |
| `RuntimeError: sizes do not match` | 尺寸不匹配 | 检查掩码尺寸 |
| `ImportError: No module named` | 缺少依赖包 | 安装missing包 |
| `cv2.error` | OpenCV错误 | 检查视频格式 |

## 获取更多帮助

如果以上FAQ没有解决您的问题，可以通过以下渠道获取帮助：

### 官方渠道
- **GitHub Issues**: [提交Bug报告](https://github.com/sczhou/ProPainter/issues)
- **GitHub Discussions**: [技术讨论](https://github.com/sczhou/ProPainter/discussions)
- **项目主页**: [官方文档](https://shangchenzhou.com/projects/ProPainter/)

### 社区资源
- **论文**: [ICCV 2023论文](https://arxiv.org/abs/2309.03897)
- **在线演示**: [Hugging Face](https://huggingface.co/spaces/sczhou/ProPainter)
- **视频教程**: [YouTube演示](https://youtu.be/92EHfgCO5-Q)

### 提问技巧

为了更快获得帮助，请在提问时包含：

1. **详细的错误信息**
2. **完整的命令行参数**
3. **系统环境信息**:
   ```bash
   python scripts/check_system.py > system_info.txt
   ```
4. **最小复现示例**
5. **期望的结果描述**

---

💡 **提示**: 建议先查阅[故障排除指南](troubleshooting.md)，其中包含更详细的诊断步骤和解决方案。