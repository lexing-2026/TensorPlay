<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo.png" alt="TensorPlay" width="400">
    </p>

<!-- 语言切换 -->
<p>
    <a href="./README.md">
        <img src="https://img.shields.io/badge/English-🇺🇸-grey?style=flat-square" alt="English">
    </a>
    <a href="./README.zh.md">
        <img src="https://img.shields.io/badge/中文-🇨🇳-yellow?style=flat-square" alt="中文">
    </a>
</p>

<!-- 平台与构建 -->
<p>
    <img src="https://img.shields.io/badge/python-3.9~3.13-blue?logo=python&logoColor=white" alt="Python Versions">
    <img src="https://img.shields.io/badge/platform-Win%20|%20Linux-purple" alt="Platform">
    <img src="https://img.shields.io/badge/hardware-CPU%20|%20CUDA%2012.x%20|%2013.x-green?logo=nvidia" alt="Hardware">
</p>

<!-- 包管理与统计 -->
<p>
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-green?logo=apache" alt="License">
    </a>
    <a href="https://pypi.org/project/tensorplay/">
        <img src="https://img.shields.io/pypi/v/tensorplay?color=blue&label=PyPI&logo=pypi" alt="PyPI Version">
    </a>
    <a href="https://pepy.tech/projects/tensorplay">
        <img src="https://static.pepy.tech/badge/TensorPlay/month" alt="Monthly Downloads">
    </a>
    <img src="https://img.shields.io/github/downloads/bluemoon-o2/TensorPlay/total.svg?label=Github%20Downloads" alt="Monthly Downloads">
</p>

<!-- 社区与支持 -->
<p>
    <a href="https://github.com/bluemoon-o2/TensorPlay/stargazers">
        <img src="https://img.shields.io/github/stars/bluemoon-o2/TensorPlay?style=flat&logo=github&color=yellow" alt="GitHub Stars">
    </a>
    <a href="https://github.com/bluemoon-o2/TensorPlay/commits/main">
        <img src="https://img.shields.io/github/last-commit/bluemoon-o2/TensorPlay?logo=git&color=teal" alt="Last Commit">
    </a>
    <a href="https://discord.gg/u6T5e2kGJm">
        <img src="https://img.shields.io/discord/1467167983616000062?color=5865F2&label=Discord&logo=discord&logoColor=white" alt="Discord">
    </a>
    <a href="https://www.tensorplay.cn/">
        <img src="https://img.shields.io/badge/Docs-tensorplay.cn-blue?logo=readthedocs" alt="Documentation">
    </a>
</p>

<h3>
    <samp>一个面向学习者的 DIY 友好型深度学习框架<br>
    旨在揭示神经网络内部机制并促进自定义硬件实验</samp>
</h3>

<p>
    <a href="https://www.tensorplay.cn/zh/guide/tutorials"><strong>🎓 教程</strong></a> •
    <a href="https://www.tensorplay.cn/"><strong>📚 文档</strong></a> •
    <a href="#quick-install"><strong>🚀 快速开始</strong></a> •
    <a href="#why-tensorplay"><strong>💡 为什么选择？</strong></a>
</p>
</div>

---

<a name="why-tensorplay"></a>
## 💡 为什么选择 TensorPlay？

TensorPlay 以**透明架构**为设计哲学，让学习者能够追踪从 Python 到 C++ 核心的每一个操作，而不会迷失在抽象层中。

<table>
<tr>
<td width="50%">

#### 🔍 纯粹且透明  
清晰可读的代码实现，允许你深入理解每个算子的底层逻辑——从自动微分到内存管理，没有黑盒。

</td>
<td width="50%">

#### 🛠️ DIY 硬件加速  
简化的 CPU 和 CUDA 后端实现，是实验自定义硬件内核、理解并行计算原理的完美游乐场。

</td>
</tr>
<tr>
<td width="50%">

#### 🧬 模块化自动微分  
通过解耦的 **TPX** 引擎，显式构建计算图，轻松理解反向传播的魔力，且易于扩展和修改。

</td>
<td width="50%">

#### 🧪 研究就绪  
高度可扩展的设计，让你能够以极少的样板代码原型化新的层类型、优化器和存储格式。

</td>
</tr>
</table>

<a name="quick-install"></a>
## 🚀 快速安装

选择适合你的安装方式：

### 📦 CPU 版本
```bash
pip install tensorplay --upgrade
```

### 🎮 CUDA 版本
```bash
# CUDA 13.0
pip install tensorplay --index-url https://download.tensorplay.cn/whl/cu130/
```
> **注意：** 确保 Python 版本与 wheel 标签匹配（如 `cp310` 对应 Python 3.10）。若遇连接问题，请确认可访问上述 URL。

### 🔧 开发安装
```bash
git clone https://github.com/bluemoon-o2/TensorPlay.git
cd TensorPlay
pip install -e .
```

## 🏗️ 架构：四大支柱

TensorPlay 建立在四个解耦的核心库之上，既可协同工作，也可独立使用：

| 库 | 核心职责 | 设计理念 |
|:-------:|:---------|:---------|
| **P10** | 🔧 核心引擎 | 提供**干净、可读**的内存管理和基础张量内核实现，是计算引擎的基石 |
| **TPX** | 🔄 自动微分 | **显式**自动微分层，让你理解或修改计算图的构建方式，完全透明 |
| **Stax** | ⚡ JIT 与优化 | 在简化环境中试验**算子融合**和**静态图捕获**，纯粹的优化游乐场 |
| **NN** | 🧩 高层 API | 与 PyTorch 兼容的模块化业务层；Linear/Conv2d 等组件可作为自定义层的蓝图 |

## 🎯 核心特性

### 📐 张量运算
- **完整自动微分**：基于 `requires_grad` 的自动梯度计算，完全对齐 PyTorch 行为
- **广播机制**：NumPy 兼容的张量广播
- **激活函数**：ReLU、Sigmoid、Tanh、Softmax、GELU 等常用激活函数
- **设备管理**：无缝 CPU/CUDA 切换，显式控制内存位置

### 🧠 神经网络层
- **线性层/全连接层**：支持权重初始化策略
- **Conv2d**：二维卷积层，理解参数计算和感受野
- **模块系统**：继承 `tp.nn.Module`，自动参数注册和结构可视化
- **损失函数**：MSE、NLL、CrossEntropy、SSE 等

### ⚙️ 优化与数据
- **优化器**：SGD、Adam、AdamW，支持学习率调度和权重衰减
- **数据加载器**：多 worker 批处理、预取 (prefetch)、自动打乱
- **早停机制**：内置早停回调，防止过拟合

## 🎓 学习路径

跟随我们的结构化教程，从零开始掌握深度学习原理：

### 初学者
1. **[从零开始的线性回归](https://www.tensorplay.cn/zh/guide/tutorials)** - 理解 `requires_grad` 和反向传播基础
2. **[MNIST CNN 图像分类](https://www.tensorplay.cn/zh/guide/tutorials)** - 使用 `Conv2d`、`MaxPool`、`DataLoader` 构建第一个神经网络

### 进阶
3. **自定义数据集与转换** - 掌握 `Dataset` 类和数据预处理流水线
4. **模型保存与加载** - 使用 `tp.save()` / `tp.load()` 和 `state_dict` 管理训练状态

👉 查看完整教程：[tutorials](https://www.tensorplay.cn/zh/guide/tutorials)

## ⚡ 快速示例

### 自动微分
```python
import tensorplay as tp

# 创建可训练张量
x = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = tp.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# 前向传播 + 反向传播
z = x.matmul(y) + tp.ones_like(x)
loss = z.sum()
loss.backward()

# 查看梯度（与 PyTorch 行为一致）
print(x.grad)  # [[6., 6.], [6., 6.]]
```

### 定义神经网络
```python
import tensorplay as tp
from tensorplay.nn import Module, Linear, ReLU, Sigmoid

class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = Linear(input_dim, hidden_dim)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_dim, output_dim)
        self.sigmoid = Sigmoid()
    
    def forward(self, x: tp.Tensor) -> tp.Tensor:
        x = self.relu(self.fc1(x))
        return self.sigmoid(self.fc2(x))

# 初始化并查看结构
model = MLP(10, 32, 1)
print(model)  # 自动生成层结构可视化
```

### 训练循环
```python
from tensorplay.data import DataLoader, TensorDataset

# 准备数据
train_data = TensorDataset(tp.randn(100, 10), tp.randn(100, 1))
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

# 训练迭代（与 PyTorch API 几乎一致）
for batch_x, batch_y in train_loader:
    predictions = model(batch_x)
    # ... 计算损失并反向传播
```

## 📊 基准测试

我们在标准数据集上提供了详细的性能对比，展示 TensorPlay 在小规模实验中的效率。查看完整的 [Benchmark Report](./benchmark/)。

<div align="center">
    <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo-footer.png" alt="TensorPlay">
</div>

## 📄 许可证

本项目采用 [Apache 2.0 许可证](LICENSE)。

## 🤝 贡献指南

我们欢迎各种形式的贡献！无论是修复 bug、改进文档，还是提出新功能建议。

### 👥 贡献者

<a href="https://github.com/bluemoon-o2/TensorPlay/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bluemoon-o2/TensorPlay&columns=10" alt="Contributors" />
</a>

## ⭐ Star 历史

<a href="https://star-history.com/#bluemoon-o2/TensorPlay&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date" width="100%" />
  </picture>
</a>

<div align="center">
    <sub>Built with ❤️ for the AI Learning Community • <a href="https://www.tensorplay.cn">TensorPlay AI</a></sub>
</div>
