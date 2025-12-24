<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo.png" alt="TensorPlay">
    </p>

[![stars](https://img.shields.io/github/stars/TensorPlay?color=ccf)](https://github.com/bluemoon-o2/TensorPlay)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/tensorplay?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLUE&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/tensorplay)
![PyPI - Downloads](https://img.shields.io/pypi/dm/TensorPlay?labelColor=blue)
![GitHub last commit](https://img.shields.io/github/last-commit/bluemoon-o2/TensorPlay?labelColor=teal)

![python](https://img.shields.io/badge/python-3.9~3.13-aff.svg)
![os](https://img.shields.io/badge/os-win-pink.svg)
![hardware](https://img.shields.io/badge/hardware-cpu%2C%20cuda-red.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/bluemoon-o2/TensorPlay)

**TensorPlay is a learner-friendly, DIY-ready deep learning framework designed for educational purposes and small-scale experiments.**
</div>

# TensorPlay
[![Docs](https://img.shields.io/badge/Docs-1.0.0rc0-blue)](https://www.tensorplay.cn/)

**TensorPlay** provides basic building blocks for constructing and training neural networks, including tensor operations, layers, optimizers, and training utilities.

## 🚀 Quick Install

### CPU Version (PyPI)
```bash
pip install tensorplay --upgrade
```

### CUDA Version (Custom Index)
For CUDA-enabled versions (e.g., CUDA 13.0), use our custom index URL:
```bash
pip install tensorplay --index-url https://download.tensorplay.cn/whl/cu130/
```
> **Note:** Ensure your Python version matches the available wheels (e.g., Python 3.10 for `cp310` wheels). Use `python --version` to check.
> If you encounter connection issues, ensure you can access the URL in your browser.

### Source Installation
```bash
git clone https://github.com/bluemoon-o2/TensorPlay.git
cd TensorPlay
pip install -e .
```

## 🏗️ Architecture: The Four Pillars
TensorPlay is built on four decoupled core libraries, each with a specific responsibility:

1. **P10 (Core)**: The foundational tensor computation engine. Handles memory, devices (CPU/CUDA), and high-performance kernels.
2. **TPX (Autograd)**: A lightweight, explicit automatic differentiation layer. Decoupled from the calculation engine for maximum transparency.
3. **Stax (JIT)**: A static graph accelerator and optimization playground for operator fusion and graph capture.
4. **NN (API)**: A modular, PyTorch-compatible high-level API providing layers, optimizers, and loss functions.

## 🎯 Core Features

| Module | Key Features                                                                                                                                      |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tensorplay.Tensor`             | Automatic gradient computation (autograd), basic arithmetic ops, broadcasting, and activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU) |
| `tensorplay.nn`      | Modular neural network layers (Linear/Dense, Conv2d), Module base class for custom models, and loss functions (MSE, NLL, CrossEntropy) |
| `tensorplay.optim`      | Optimization algorithms (Adam, SGD, AdamW) with parameter update logic |
| `tensorplay.data`       | DataLoader for batching, shuffling, and data iteration                                                              |

## ⚡Basic Usage
TensorPlay’s API is intentionally designed to match PyTorch, so if you know PyTorch, you already know most of TensorPlay!

### 1. Tensors and Automatic Differentiation
```python
import tensorplay as tp

# Create a tensor with requires_grad=True (for autograd)
x = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = tp.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

# Perform operations
z = x.matmul(y) + tp.ones_like(x)
loss = z.sum()

# Backpropagation (compute gradients)
loss.backward()

# Inspect gradients (aligned with PyTorch's grad behavior)
print(x.grad)  # Gradient of loss w.r.t. x
print(y.grad)  # Gradient of loss w.r.t. y
```
### 2. Define a Neural Network
Inherit from tp.nn.Module (just like torch.nn.Module) and define the forward pass:
```python
import tensorplay as tp
from tensorplay.nn import Module, Linear, ReLU, Sigmoid

class MLP(Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        # Linear layer (alias: Dense, same as torch.nn.Linear)
        self.fc1 = Linear(input_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        # Activation functions (functional or module-based)
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

    def forward(self, x: tp.Tensor) -> tp.Tensor:
        # Forward pass (exactly like PyTorch)
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

# Initialize the model
model = MLP(input_dim=10, hidden_dim=32, output_dim=1)
# Print model structure (automatically implemented by Module)
print(model)
```
### 3. Prepare Data with DataLoader
Train the model using the `train_on_batch` function. A typical training loop includes batch training, validation, and early stopping judgment:
```python
import tensorplay as tp
from tensorplay.data import DataLoader, TensorDataset

# Sample data (features: (N, 10), labels: (N, 1))
train_data = TensorDataset(tp.randn(100, 10), tp.randn(100, 1))

# DataLoader (batch_size, shuffle)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=8,
    num_workers=2,
    prefetch_factor=2,
    shuffle=True,
    drop_last=False
)
```
### 4. Evaluate the Model
```python
def test(model, test_loader):
    correct = 0
    total = 0
    for batch_x, batch_y in test_loader:
        for x, y in zip(batch_x, batch_y):
            # Make predictions on a single sample
            # Set the threshold according to the task type 
            # (taking binary classification as an example here)
            pred = 1 if model(x).item() > 0.5 else 0  # For binary classification
            if pred == y.item():
                correct += 1
            total += 1
    print(f"Test Accuracy: {correct/total:.4f}")
```

## ♨️ Benchmark
We provide benchmark results on several standard datasets to demonstrate TensorPlay's performance and usability. Detailed benchmark results and comparisons with other frameworks can be found in the [Benchmark Report](./benchmark/).

<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/logo.png" alt="TensorPlay">
    </p>
</div>

## 👩‍👩‍👧‍👦Contributing
Contributions are welcome! Feel free to open issues for bugs or feature requests, or submit pull requests with improvements.

## 📄 License
This project is released under the [Apache 2.0 license](LICENSE).  

## 🔗Links
* [Source code and issue tracker](https://github.com/bluemoon-o2/TensorPlay)
* [PyPI release](https://pypi.org/project/TensorPlay/)
* [Documentation](https://www.welog.me/article/TensorPLay)