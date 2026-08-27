<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/tensorplay-mark.png" alt="TensorPlay 图形标志" width="180">
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
    <img src="https://img.shields.io/github/downloads/bluemoon-o2/TensorPlay/total.svg?label=Github%20Downloads" alt="Github Downloads">
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
    <a href="https://www.tensorplay.cn/zh/guide/tutorials"><strong>教程</strong></a> •
    <a href="https://www.tensorplay.cn/"><strong>文档</strong></a> •
    <a href="#getting-started"><strong>快速开始</strong></a> •
    <a href="#installation"><strong>安装指南</strong></a>
</p>
</div>

--------------------------------------------------------------------------------

TensorPlay 是一个 Python 软件包，提供两大高层能力：

- 带有强大 GPU 加速的张量计算（类 NumPy）
- 建立在显式、基于磁带（tape）机制的自动微分系统之上的深度神经网络

整个技术栈——Python API、C++ 核心、CUDA 内核、编译器——都为「可读」而生：实现干净、计算图显式，模型与硬件之间没有黑盒。

<!-- toc -->

- [关于 TensorPlay](#关于-tensorplay)
  - [一个透明的张量库](#一个透明的张量库)
  - [为什么选择 TensorPlay](#为什么选择-tensorplay)
- [安装](#安装)
  - [二进制安装](#二进制安装)
  - [从源码构建](#从源码构建)
    - [环境要求](#环境要求)
    - [获取源码](#获取源码)
    - [安装构建依赖](#安装构建依赖)
    - [安装 TensorPlay](#安装-tensorplay)
    - [调整构建选项（可选）](#调整构建选项可选)
- [快速上手](#快速上手)
  - [自动微分](#自动微分)
  - [定义神经网络](#定义神经网络)
  - [训练循环](#训练循环)
- [测试](#测试)
- [资源](#资源)
- [发布与贡献](#发布与贡献)
- [许可证](#许可证)

<!-- tocstop -->

## 关于 TensorPlay

### 一个透明的张量库

在更细的粒度上，TensorPlay 由以下组件构成：

| 组件 | 说明 |
| ---- | ---- |
| **tensorplay** | Python API：张量、自动微分、`nn`、`optim`、数据加载、序列化——对外公开层 |
| **p10** | C++ 核心引擎：张量存储与内存管理、基础的 CPU 与 CUDA 内核实现 |
| **tpx** | 自动微分层：显式构建计算图并执行反向传播，与核心完全解耦 |
| **stax** | JIT 编译器试验场：静态图捕获与算子融合实验，支持自定义算子的原生下沉 |
| **tensorplay.nn** | 神经网络构件：`Module`、`Linear`、`Conv2d`、激活函数、损失函数与容器抽象 |
| **tensorplay.optim** | 优化器（SGD、Adam、AdamW），支持学习率调度与权重衰减 |
| **tensorplay.data** | `Dataset` / `DataLoader`：多 worker 批处理、预取 (prefetch) 与自动打乱 |
| **tensorplay.library** | 一等公民的自定义算子：注册算子、挂接 fake/meta 与自动微分公式、接入自有 Triton kernel |

四大支柱——**P10**、**TPX**、**Stax** 与 **NN**——是刻意解耦的核心库，既可协同工作，也可独立使用。`linalg`、`fft`、`sparse`、`special`、`amp`、`distributed`、`serialization` 等领域子包补全了完整的 API 面。

### 为什么选择 TensorPlay

TensorPlay 以**透明架构**为设计哲学：每个操作都能从 Python 追踪到 C++ 核心，而不会迷失在抽象层中。

- **纯粹且可读的实现。** 深入理解每个算子的底层逻辑——从自动微分到内存管理，没有黑盒。
- **DIY 硬件加速。** 简化的 CPU/CUDA 后端是实验自定义硬件内核、学习并行计算原理的游乐场。
- **模块化自动微分。** 解耦的 TPX 引擎显式构建计算图，反向传播的原理一目了然，且易于扩展。
- **研究就绪。** 以极少的样板代码原型化新的层类型、优化器、自定义算子与存储格式。
- **亲切的 API 风格。** 用过主流深度学习框架的话，心智模型可以直接迁移——把时间花在理解内部机制上，而不是语法上。

## 安装

### 二进制安装

```bash
# 从 PyPI 安装 CPU 版本
pip install tensorplay --upgrade

# 从 TensorPlay CUDA 源安装 CUDA 版本（可选 cu124、cu126 或 cu130）
# PyPI 作为运行时依赖的额外索引
pip install tensorplay --index-url https://download.tensorplay.cn/whl/cu124/ --extra-index-url https://pypi.org/simple
```

> **注意：** 请确保 Python 版本与 wheel 标签匹配（如 `cp310` 对应 Python 3.10）。CUDA 版本要求驱动与运行时支持对应 CUDA 版本。

### 从源码构建

源码构建得到的是可魔改、可调试的安装——推荐贡献者以及所有研究内核的开发者使用。

#### 环境要求

- Python >= 3.9，< 3.14
- CMake >= 3.18（< 4.0）
- 支持 C++20 的编译器（Windows 使用 MSVC 2022，Linux 使用 GCC/Clang）
- CUDA Toolkit（可选，用于 GPU 支持）；可通过 `CMAKE_CUDA_ARCHITECTURES` 指定目标 GPU 架构
- [Ninja](https://ninja-build.org/)（随构建依赖自动安装）

#### 获取源码

```bash
git clone https://github.com/bluemoon-o2/TensorPlay.git
cd TensorPlay
```

#### 安装构建依赖

隔离的 PEP 517 构建会自动获取全部依赖。若需更快的迭代开发，可先装好工具链再关闭隔离：

```bash
pip install -r requirements-build.txt
```

#### 安装 TensorPlay

TensorPlay 通过 `pyproject.toml` 中声明的标准 PEP 517 接口由 scikit-build-core 构建：

```bash
# 完整构建并安装（加 -v 可查看详细输出）
pip install .

# 开发用可编辑安装
pip install -e . --no-build-isolation

# 或只产出 wheel 不安装
python -m build --wheel
```

构建成功后，`import tensorplay` 会加载安装包内编译好的 `_C` 扩展。

#### 调整构建选项（可选）

构建由环境变量驱动——无需 `-D` 参数：

```bash
# 仅构建 CPU 版本
USE_CUDA=OFF pip install .

# 指定目标 GPU 架构
CMAKE_CUDA_ARCHITECTURES="70;75;86" pip install .
```

| 变量 | 默认值 | 说明 |
| ---- | ---- | ---- |
| `USE_CUDA` | 自动探测 | 启用/禁用 CUDA 构建 |
| `BUILD_TESTS` | `OFF` | 构建 C++ 测试套件 |
| `USE_BLAS` / `USE_ONEDNN` | `ON` | BLAS 加速 / oneDNN 算子库 |
| `MAX_JOBS` | 机器默认 | 限制编译并行度 |
| `DEBUG` | 未设置 | 以 `-O0 -g` 构建 |
| `CMAKE_CUDA_ARCHITECTURES` | 本机架构 | 分号分隔的 GPU 架构列表 |
| `TENSORPLAY_BUILD_VERSION` / `TENSORPLAY_BUILD_NUMBER` | 未设置 | 覆盖包版本号（发布 CI 使用） |

所有 `USE_*`、`BUILD_*`、`CMAKE_*` 环境变量都会自动转发给 CMake——无需任何额外参数。

## 快速上手

### 自动微分

```python
import tensorplay as tp

x = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = tp.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

z = x.matmul(y) + tp.ones_like(x)
loss = z.sum()
loss.backward()

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

model = MLP(10, 32, 1)
print(model)  # 自动生成层结构可视化
```

### 训练循环

```python
from tensorplay.data import DataLoader, TensorDataset

train_data = TensorDataset(tp.randn(100, 10), tp.randn(100, 1))
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

for batch_x, batch_y in train_loader:
    predictions = model(batch_x)
    # ... 计算损失、调用 loss.backward()、更新优化器
```

体系化教程——从零开始的线性回归、MNIST CNN 图像分类、自定义数据集、`.mega` + `state_dict` 模型保存与加载——见 [tensorplay.cn](https://www.tensorplay.cn/zh/guide/tutorials)。

## 测试

Python 测试套件使用 pytest 针对已安装（或原地构建）的包运行：

```bash
pytest test/
```

CI 会在每个 PR 和 main 推送上构建完整 wheel 矩阵（Python 3.9–3.13；CPU 覆盖 Linux x86_64/aarch64、macOS arm64、Windows x86_64；CUDA 覆盖 Linux 和 Windows x86_64）并验证每个 wheel；参见 [.github/workflows/](.github/workflows/)（`pull`、`trunk`、`publish`、`lint`）。Lint 规则位于 `pyproject.toml` 的 `[tool.ruff]` 段。

## 资源

- **文档：** [tensorplay.cn](https://www.tensorplay.cn/)
- **教程：** [tensorplay.cn/zh/guide/tutorials](https://www.tensorplay.cn/zh/guide/tutorials)
- **基准测试：** [benchmark/](benchmark/)
- **社区：** [Discord](https://discord.gg/u6T5e2kGJm)

## 发布与贡献

包版本号以 [`version.txt`](version.txt) 为单一来源（开发安装会带 `+git<sha>` 后缀）。正式发布通过推送 `v主版本.次版本.0` tag 触发；例如 `v1.2.1` 这样的修订版本 tag 不会发布二进制包。CI 流水线自动构建并验证完整 wheel 矩阵，CPU wheel 发布到 PyPI，CUDA wheel 上传到 GitHub Release，并将其 PEP 503 索引部署到 Cloudflare Pages。

发布工作流使用 GitHub 的 `pypi` 环境进行 PyPI 可信发布。Python 版本以及 CPU/CUDA 平台 runner 配置在 [`.github/wheel-platforms.json`](.github/wheel-platforms.json) 中；当前 CUDA 版本为 `cu124`、`cu126`、`cu130`，配置在 [`.github/cuda-variants.json`](.github/cuda-variants.json) 中。未来增加 `{ "variant": "cu132", "toolkit": "13.2.1" }` 后，CI 会构建并在同一次 Pages 部署中发布新的 CUDA 索引。CUDA Release wheel 会带 `+cuXXX` 本地版本后缀，确保不同 CUDA 版本可以共存于同一个 GitHub Release。CUDA 索引部署需要仓库 Secrets `CLOUDFLARE_API_TOKEN` 和 `CLOUDFLARE_ACCOUNT_ID`；可选的仓库变量 `CLOUDFLARE_PAGES_PROJECT_NAME` 默认为 `tensorplay-pypi`。

我们欢迎各种形式的贡献——bug 修复、文档改进、新功能建议。开发流程与编码规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

<a href="https://github.com/bluemoon-o2/TensorPlay/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bluemoon-o2/TensorPlay&columns=10" alt="Contributors" />
</a>

## 许可证

TensorPlay 采用 [Apache 2.0 许可证](LICENSE)。

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
