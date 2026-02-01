<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo.png" alt="TensorPlay" width="400">
    </p>

<!-- Language Switch -->
<p>
    <a href="./README.md">
        <img src="https://img.shields.io/badge/English-🇺🇸-yellow?style=flat-square" alt="English">
    </a>
    <a href="./README.zh.md">
        <img src="https://img.shields.io/badge/中文-🇨🇳-blue?style=flat-square" alt="中文">
    </a>
</p>

<!-- Platform & Build -->
<p>
    <img src="https://img.shields.io/badge/python-3.9~3.13-blue?logo=python&logoColor=white" alt="Python Versions">
    <img src="https://img.shields.io/badge/platform-Win%20|%20Linux-purple" alt="Platform">
    <img src="https://img.shields.io/badge/hardware-CPU%20|%20CUDA%2012.x%20|%2013.x-green?logo=nvidia" alt="Hardware">
</p>

<!-- Package & Stats -->
<p>
    <a href="https://pypi.org/project/tensorplay/">
        <img src="https://img.shields.io/pypi/v/tensorplay?color=blue&label=PyPI&logo=pypi" alt="PyPI Version">
    </a>
    <a href="https://pepy.tech/projects/tensorplay">
        <img src="https://static.pepy.tech/personalized-badge/tensorplay?period=total&units=INTERNATIONAL_SYSTEM&left_color=grey&right_color=blue&left_text=downloads" alt="Downloads">
    </a>
    <img src="https://img.shields.io/pypi/dm/TensorPlay?label=monthly%20downloads" alt="Monthly Downloads">
</p>

<!-- Community -->
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
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-green?logo=apache" alt="License">
    </a>
</p>

<h3>
    <samp>A learner-friendly, DIY-ready deep learning framework<br>
    designed to reveal neural network internals and facilitate custom hardware experimentation.</samp>
</h3>

<p>
    <a href="https://www.tensorplay.cn/en/guide/tutorials"><strong>🎓 Tutorials</strong></a> •
    <a href="https://www.tensorplay.cn/"><strong>📚 Docs</strong></a> •
    <a href="#-quick-install"><strong>🚀 Quick Start</strong></a> •
    <a href="#-why-tensorplay"><strong>💡 Why TensorPlay?</strong></a>
</p>
</div>


## 💡 Why TensorPlay?

TensorPlay is built on the **philosophy of transparency**, allowing learners to trace every operation from Python to C++ core without getting lost in abstraction layers.

<table>
<tr>
<td width="50%">

#### 🔍 Pure & Transparent  
Clean, readable implementations that let you dive deep into the logic of every operator—from autograd to memory management. No black boxes.

</td>
<td width="50%">

#### 🛠️ DIY Acceleration  
Simplified CPU and CUDA backend implementations serve as the perfect playground for experimenting with custom hardware kernels and understanding parallel computing principles.

</td>
</tr>
<tr>
<td width="50%">

#### 🧬 Modular Autograd  
Through the decoupled **TPX** engine, computation graphs are built explicitly, making it easy to understand the magic of backpropagation and simple to extend.

</td>
<td width="50%">

#### 🧪 Research Ready  
Highly extensible design allows you to prototype new layer types, optimizers, and storage formats with minimal boilerplate code.

</td>
</tr>
</table>

## 🚀 Quick Install

Choose your installation method:

### 📦 CPU Version
```bash
pip install tensorplay --upgrade
```

### 🎮 CUDA Version
```bash
# CUDA 13.0
pip install tensorplay --index-url https://download.tensorplay.cn/whl/cu130/
```
> **Note:** Ensure your Python version matches the wheel tags (e.g., `cp310` for Python 3.10). If you encounter connection issues, please verify access to the above URLs.

### 🔧 Development Install
```bash
git clone https://github.com/bluemoon-o2/TensorPlay.git
cd TensorPlay
pip install -e .
```

## 🏗️ Architecture: The Four Pillars

TensorPlay is built upon four decoupled core libraries that can work together or independently:

| Library | Core Responsibility | Design Philosophy |
|:-------:|:-------------------|:------------------|
| **P10** | 🔧 Core Engine | Provides **clean, readable** memory management and foundational tensor kernel implementations—the cornerstone of the computation engine |
| **TPX** | 🔄 Autograd | **Explicit** automatic differentiation layer that lets you understand or modify how computation graphs are built, completely transparent |
| **Stax** | ⚡ JIT & Optimization | Experiment with **operator fusion** and **static graph capture** in a simplified environment—a pure optimization playground |
| **NN** | 🧩 High-level API | PyTorch-compatible modular business layer; components like Linear/Conv2d serve as blueprints for custom layers |

## 🎯 Core Features

### 📐 Tensor Operations
- **Full Autograd**: Automatic gradient computation based on `requires_grad`, fully aligned with PyTorch behavior
- **Broadcasting**: NumPy-compatible tensor broadcasting mechanisms
- **Activations**: ReLU, Sigmoid, Tanh, Softmax, GELU, and other common activation functions
- **Device Management**: Seamless CPU/CUDA switching with explicit memory location control

### 🧠 Neural Network Layers
- **Linear/Dense**: Fully connected layers with weight initialization strategies
- **Conv2d**: 2D convolutional layers for understanding parameter calculation and receptive fields
- **Module System**: Inherit from `tp.nn.Module` for automatic parameter registration and architecture visualization
- **Loss Functions**: MSE, NLL, CrossEntropy, SSE, etc.

### ⚙️ Optimization & Data
- **Optimizers**: SGD, Adam, AdamW with learning rate scheduling and weight decay support
- **DataLoader**: Multi-worker batch processing, prefetching, and automatic shuffling
- **Early Stopping**: Built-in early stopping callback to prevent overfitting

## 🎓 Learning Path

Follow our structured tutorials to master deep learning principles from scratch:

### Beginners
1. **[Linear Regression from Scratch](https://www.tensorplay.cn/en/guide/tutorials)** - Understand `requires_grad` and backpropagation fundamentals
2. **[MNIST CNN Image Classification](https://www.tensorplay.cn/en/guide/tutorials)** - Build your first neural network using `Conv2d`, `MaxPool`, and `DataLoader`

### Advanced
3. **Custom Datasets & Transforms** - Master the `Dataset` class and data preprocessing pipelines
4. **Model Saving & Loading** - Use `tp.save()` / `tp.load()` and `state_dict` to manage training states

👉 View full tutorials: [tutorials](https://www.tensorplay.cn/en/guide/tutorials)

## ⚡ Quick Examples

### Automatic Differentiation
```python
import tensorplay as tp

# Create trainable tensors
x = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = tp.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Forward pass + Backward pass
z = x.matmul(y) + tp.ones_like(x)
loss = z.sum()
loss.backward()

# View gradients (consistent with PyTorch behavior)
print(x.grad)  # [[6., 6.], [6., 6.]]
```

### Define a Neural Network
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

# Initialize and view structure
model = MLP(10, 32, 1)
print(model)  # Auto-generated architecture visualization
```

### Training Loop
```python
from tensorplay.data import DataLoader, TensorDataset

# Prepare data
train_data = TensorDataset(tp.randn(100, 10), tp.randn(100, 1))
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

# Training iteration (API almost identical to PyTorch)
for batch_x, batch_y in train_loader:
    predictions = model(batch_x)
    # ... compute loss and backpropagate
```

## 📊 Benchmarks

We provide detailed performance comparisons on standard datasets, demonstrating TensorPlay's efficiency in small-scale experiments. View the complete [Benchmark Report](./benchmark/).

<div align="center">
    <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo-footer.png" alt="TensorPlay">
</div>

## 📄 License

This project is licensed under the [Apache 2.0 License](LICENSE).

## 🤝 Contributing

We welcome contributions in all forms! Whether it's bug fixes, documentation improvements, or new feature suggestions.

### 👥 Contributors

<a href="https://github.com/bluemoon-o2/TensorPlay/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bluemoon-o2/TensorPlay&columns=10" alt="Contributors" />
</a>

## ⭐ Star History

<a href="https://star-history.com/#bluemoon-o2/TensorPlay&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=bluemoon-o2/TensorPlay&type=Date" width="100%" />
  </picture>
</a>

<div align="center">
    <sub>Built with ❤️ for the AI Learning Community • <a href="https://www.tensorplay.cn">Official Website</a></sub>
</div>
