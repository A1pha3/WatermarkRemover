# 贡献指南

欢迎为ProPainter项目做出贡献！本文档将指导您如何参与项目开发和改进。

## 贡献方式

### 代码贡献
- 修复Bug
- 添加新功能
- 性能优化
- 代码重构

### 文档贡献
- 改进现有文档
- 添加使用示例
- 翻译文档
- 编写教程

### 测试贡献
- 报告Bug
- 提供测试用例
- 性能基准测试
- 兼容性测试

### 社区贡献
- 回答问题
- 分享经验
- 推广项目
- 组织活动

## 开发环境设置

### 1. Fork和Clone项目

```bash
# Fork项目到您的GitHub账户
# 然后clone到本地
git clone https://github.com/YOUR_USERNAME/ProPainter.git
cd ProPainter

# 添加上游仓库
git remote add upstream https://github.com/sczhou/ProPainter.git
```

### 2. 创建开发环境

```bash
# 创建开发环境
conda create -n propainter-dev python=3.8
conda activate propainter-dev

# 安装依赖
pip install -r requirements.txt

# 安装开发工具
pip install pytest black flake8 pre-commit
```

### 3. 安装Pre-commit钩子

```bash
# 安装pre-commit钩子
pre-commit install

# 运行所有文件的检查
pre-commit run --all-files
```

## 开发流程

### 1. 创建功能分支

```bash
# 同步最新代码
git fetch upstream
git checkout main
git merge upstream/main

# 创建功能分支
git checkout -b feature/your-feature-name
```

### 2. 编写代码

遵循以下编码规范：

#### Python代码规范
- 使用PEP 8风格
- 函数和变量使用snake_case
- 类名使用PascalCase
- 常量使用UPPER_CASE

```python
# 好的示例
class VideoProcessor:
    """视频处理器类"""
    
    DEFAULT_FPS = 30
    
    def __init__(self, input_path: str):
        self.input_path = input_path
        self._frame_count = 0
    
    def process_video(self, output_path: str) -> bool:
        """处理视频文件
        
        Args:
            output_path: 输出文件路径
            
        Returns:
            bool: 处理是否成功
        """
        # 实现逻辑
        return True
```

#### 注释规范
- 使用英文注释
- 函数必须有docstring
- 复杂逻辑需要行内注释

### 3. 编写测试

为新功能编写测试用例：

```python
# tests/test_video_processor.py
import pytest
from core.video_processor import VideoProcessor

class TestVideoProcessor:
    def test_init(self):
        """测试初始化"""
        processor = VideoProcessor("test.mp4")
        assert processor.input_path == "test.mp4"
    
    def test_process_video(self):
        """测试视频处理"""
        processor = VideoProcessor("test.mp4")
        result = processor.process_video("output.mp4")
        assert result is True
```

### 4. 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_video_processor.py

# 生成覆盖率报告
pytest --cov=core --cov-report=html
```

### 5. 代码格式化

```bash
# 使用black格式化代码
black .

# 检查代码风格
flake8 .

# 类型检查
mypy core/
```

## 提交规范

### Commit消息格式

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

#### Type类型
- `feat`: 新功能
- `fix`: Bug修复
- `docs`: 文档更新
- `style`: 代码格式化
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

#### 示例
```bash
git commit -m "feat(core): add video preprocessing module

- Add support for multiple video formats
- Implement frame extraction and resizing
- Add comprehensive error handling

Closes #123"
```

## Pull Request流程

### 1. 推送代码

```bash
# 推送到您的fork仓库
git push origin feature/your-feature-name
```

### 2. 创建Pull Request

在GitHub上创建PR时，请：
- 使用清晰的标题
- 详细描述更改内容
- 关联相关的Issue
- 添加适当的标签

#### PR模板

```markdown
## 更改说明
简要描述本次更改的内容和目的。

## 更改类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 性能优化
- [ ] 代码重构

## 测试
- [ ] 已添加测试用例
- [ ] 所有测试通过
- [ ] 手动测试完成

## 检查清单
- [ ] 代码遵循项目规范
- [ ] 已更新相关文档
- [ ] 已添加必要的注释
- [ ] 无明显的性能问题

## 相关Issue
Closes #issue_number
```

### 3. 代码审查

PR提交后会进行代码审查：
- 响应审查意见
- 及时修改代码
- 保持友好沟通

## 项目结构

### 核心模块

```
ProPainter/
├── core/                   # 核心功能
│   ├── dataset.py         # 数据集处理
│   ├── loss.py           # 损失函数
│   ├── metrics.py        # 评估指标
│   └── trainer.py        # 训练器
├── model/                 # 模型定义
│   ├── propainter.py     # 主模型
│   ├── modules/          # 子模块
│   └── misc.py           # 工具函数
├── utils/                 # 工具函数
├── scripts/              # 脚本文件
├── configs/              # 配置文件
└── docs/                 # 文档
```

### 添加新模块

1. 在相应目录创建模块文件
2. 更新`__init__.py`文件
3. 添加相应的测试
4. 更新文档

## 文档贡献

### 文档结构

```
docs/
├── installation.md        # 安装指南
├── quick_start.md        # 快速开始
├── api_reference.md      # API参考
├── training.md           # 训练指南
├── evaluation.md         # 评估指南
├── troubleshooting.md    # 故障排除
└── contributing.md       # 贡献指南
```

### 文档编写规范

- 使用Markdown格式
- 保持结构清晰
- 包含代码示例
- 添加必要的图表

## Bug报告

### 报告模板

```markdown
## Bug描述
清晰简洁地描述Bug。

## 复现步骤
1. 执行命令 '...'
2. 点击 '....'
3. 滚动到 '....'
4. 看到错误

## 期望行为
描述您期望发生的情况。

## 实际行为
描述实际发生的情况。

## 环境信息
- OS: [e.g. Ubuntu 20.04]
- Python版本: [e.g. 3.8.10]
- PyTorch版本: [e.g. 1.12.1]
- CUDA版本: [e.g. 11.3]

## 附加信息
添加任何其他相关信息。
```

## 功能请求

### 请求模板

```markdown
## 功能描述
清晰地描述您希望添加的功能。

## 使用场景
描述这个功能的使用场景和价值。

## 建议实现
如果有想法，请描述您认为应该如何实现。

## 替代方案
描述您考虑过的其他解决方案。

## 附加信息
添加任何其他相关信息。
```

## 发布流程

### 版本号规范

使用语义化版本控制：
- 主版本号：不兼容的API修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修正

### 发布检查清单

- [ ] 所有测试通过
- [ ] 文档已更新
- [ ] 版本号已更新
- [ ] 更新日志已编写
- [ ] 性能基准测试完成

## 社区准则

### 行为准则

- 尊重他人
- 保持专业
- 欢迎新手
- 建设性反馈

### 沟通渠道

- **GitHub Issues**: Bug报告和功能请求
- **GitHub Discussions**: 技术讨论
- **Email**: 私人联系

## 致谢

感谢所有为ProPainter做出贡献的开发者！

### 贡献者列表

- [Shangchen Zhou](https://github.com/sczhou) - 项目创始人
- [Chongyi Li](https://github.com/li-chongyi) - 核心开发者
- 以及所有其他贡献者

### 如何成为维护者

长期贡献者可以申请成为项目维护者：
1. 持续贡献高质量代码
2. 积极参与社区讨论
3. 帮助其他贡献者
4. 展现出色的技术能力

## 资源链接

- [项目主页](https://shangchenzhou.com/projects/ProPainter/)
- [论文链接](https://arxiv.org/abs/2309.03897)
- [在线演示](https://huggingface.co/spaces/sczhou/ProPainter)

## 许可证

本项目采用[NTU S-Lab License 1.0](../LICENSE)协议。贡献代码即代表您同意将代码按此许可证发布。

## 联系方式

如有任何问题，请通过以下方式联系：
- 项目Issues: https://github.com/sczhou/ProPainter/issues
- 邮件: shangchenzhou@gmail.com

感谢您对ProPainter项目的贡献！