# ProPainter 文档中心

欢迎来到ProPainter的文档中心！这里提供了完整的使用指南和开发文档。

## 文档导航

### 🚀 快速开始
- [安装指南](installation.md) - 环境配置和依赖安装
- [快速开始](quick_start.md) - 5分钟上手指南

### 📖 用户指南
- [API参考](api_reference.md) - 完整的API接口文档
- [内存优化](memory_optimization.md) - GPU显存优化策略
- [Web演示](web_demo.md) - 在线和本地演示部署

### 🔬 高级用法
- [训练指南](training.md) - 自定义模型训练
- [评估指南](evaluation.md) - 模型性能评估
- [数据集准备](dataset_preparation.md) - 数据预处理和组织

### 🛠️ 开发者资源
- [常见问题](faq.md) - 快速问答和解决方案
- [故障排除](troubleshooting.md) - 详细诊断和修复指南
- [贡献指南](contributing.md) - 参与项目开发
- [性能优化](performance_guide.md) - 性能调优和硬件配置
- [更新日志](changelog.md) - 版本更新历史

## 项目概述

ProPainter是一个基于深度学习的先进视频修复工具，发表于ICCV 2023，主要功能包括：

### 🎯 核心功能
- **对象移除**: 智能移除视频中的不需要对象
- **视频补全**: 高质量填补视频中的缺失区域  
- **时序一致性**: 确保修复结果在时间维度上的连贯性
- **交互式编辑**: 支持用户友好的交互式对象选择

### 🏆 技术优势
- **先进算法**: 基于改进的传播机制和Transformer架构
- **高质量输出**: ICCV 2023论文验证的最先进性能
- **内存优化**: 支持不同显存配置的优化策略
- **易于使用**: 提供命令行、Web界面和API多种使用方式

## 核心特性

### 🎯 高质量修复
- 基于先进的Transformer架构
- 优化的传播机制
- 时序一致性保证

### ⚡ 高效处理
- 内存优化算法
- GPU加速推理
- 批量处理支持

### 🔧 易于使用
- 简单的命令行接口
- 交互式Web界面
- 丰富的参数配置

## 快速导航

| 我想... | 推荐文档 |
|---------|----------|
| 第一次了解ProPainter | [新手入门](getting_started.md) |
| 快速试用ProPainter | [快速开始](quick_start.md) |
| 学习使用技巧 | [最佳实践](best_practices.md) |
| 解决安装问题 | [安装指南](installation.md) + [故障排除](troubleshooting.md) |
| 了解所有参数 | [API参考](api_reference.md) |
| 优化内存使用 | [内存优化](memory_optimization.md) |
| 提升处理性能 | [性能优化](performance_guide.md) |
| 训练自定义模型 | [训练指南](training.md) + [数据集准备](dataset_preparation.md) |
| 评估模型性能 | [评估指南](evaluation.md) |
| 部署Web服务 | [Web演示](web_demo.md) |
| 贡献代码 | [贡献指南](contributing.md) |
| 查看更新历史 | [更新日志](changelog.md) |

## 版本信息

- **当前版本**: 1.0.0
- **更新日期**: 2024年10月
- **兼容性**: Python 3.8+, PyTorch 1.7.1+, CUDA 9.2+

## 获取帮助

### 在线资源
- [项目主页](https://shangchenzhou.com/projects/ProPainter/)
- [GitHub仓库](https://github.com/sczhou/ProPainter)
- [论文链接](https://arxiv.org/abs/2309.03897)

### 在线演示
- [Hugging Face Spaces](https://huggingface.co/spaces/sczhou/ProPainter)
- [OpenXLab平台](https://openxlab.org.cn/apps/detail/ShangchenZhou/ProPainter)

### 社区支持
- [GitHub Issues](https://github.com/sczhou/ProPainter/issues) - Bug报告和功能请求
- [GitHub Discussions](https://github.com/sczhou/ProPainter/discussions) - 技术讨论

## 引用

如果ProPainter对您的研究有帮助，请引用我们的论文：

```bibtex
@inproceedings{zhou2023propainter,
   title={{ProPainter}: Improving Propagation and Transformer for Video Inpainting},
   author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
   booktitle={Proceedings of IEEE International Conference on Computer Vision (ICCV)},
   year={2023}
}
```

## 许可证

本项目采用[NTU S-Lab License 1.0](../LICENSE)协议，仅限非商业用途。

---

📝 文档持续更新中，如发现问题请提交[Issue](https://github.com/sczhou/ProPainter/issues)。