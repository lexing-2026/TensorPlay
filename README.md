<div align="center">
    <p>
        <img src="https://raw.githubusercontent.com/bluemoon-o2/TensorPlay/main/docs/images/tensorplay-mark.png" alt="TensorPlay mark" width="180">
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
</p>

<h3>
    <samp>A learner-friendly, DIY-ready deep learning framework<br>
    designed to reveal neural network internals and facilitate custom hardware experimentation.</samp>
</h3>

<p>
    <a href="https://www.tensorplay.cn/en/guide/tutorials"><strong>Tutorials</strong></a> •
    <a href="https://www.tensorplay.cn/"><strong>Docs</strong></a> •
    <a href="#getting-started"><strong>Quick Start</strong></a> •
    <a href="#installation"><strong>Installation</strong></a>
</p>
</div>

--------------------------------------------------------------------------------

TensorPlay is a Python package that provides two high-level features:

- Tensor computation (like NumPy) with strong GPU acceleration
- Deep neural networks built on an explicit, tape-based autograd system

The whole stack — Python API, C++ core, CUDA kernels, compiler — is engineered to be read: clean implementations, explicit computation graphs, and no black box between your model and the hardware.

<!-- toc -->

- [About TensorPlay](#about-tensorplay)
  - [A Transparent Tensor Library](#a-transparent-tensor-library)
  - [Why TensorPlay](#why-tensorplay)
- [Installation](#installation)
  - [Binaries](#binaries)
  - [From Source](#from-source)
    - [Prerequisites](#prerequisites)
    - [Get the TensorPlay Source](#get-the-tensorplay-source)
    - [Install Build Dependencies](#install-build-dependencies)
    - [Install TensorPlay](#install-tensorplay)
    - [Adjusting Build Options (Optional)](#adjusting-build-options-optional)
- [Getting Started](#getting-started)
  - [Automatic Differentiation](#automatic-differentiation)
  - [Defining a Neural Network](#defining-a-neural-network)
  - [Training Loop](#training-loop)
- [Testing](#testing)
- [Resources](#resources)
- [Releases and Contributing](#releases-and-contributing)
- [License](#license)

<!-- tocstop -->

## About TensorPlay

### A Transparent Tensor Library

At a granular level, TensorPlay consists of the following components:

| Component | Description |
| ---- | ---- |
| **tensorplay** | The Python API: tensors, autograd, `nn`, `optim`, data loading, serialization — the public surface |
| **p10** | C++ core engine: tensor storage and memory management, foundational CPU and CUDA kernels |
| **tpx** | Autograd layer: explicit computation-graph construction and backward execution, fully decoupled from the core |
| **stax** | JIT compiler playground: static graph capture and operator fusion experiments, including native lowering of custom ops |
| **tensorplay.nn** | Neural network building blocks: `Module`, `Linear`, `Conv2d`, activations, losses, container abstractions |
| **tensorplay.optim** | Optimizers (SGD, Adam, AdamW) with learning-rate scheduling and weight decay |
| **tensorplay.data** | `Dataset` / `DataLoader` with multi-worker batching, prefetching and shuffling |
| **tensorplay.library** | First-class custom operators: register ops, attach fake/meta and autograd formulas, bring your own Triton kernels |

The four pillars — **P10**, **TPX**, **Stax** and **NN** — are deliberately decoupled libraries that can work together or independently. Domain subpackages such as `linalg`, `fft`, `sparse`, `special`, `amp`, `distributed` and `serialization` round out the API surface.

### Why TensorPlay

TensorPlay is built on a philosophy of **transparency**: you can trace every operation from Python into the C++ core without getting lost in abstraction layers.

- **Pure and readable implementations.** Dive into the logic of every operator — from autograd to memory management. No black boxes.
- **DIY acceleration.** Simplified CPU and CUDA backends are a playground for experimenting with custom hardware kernels and learning parallel computing.
- **Modular autograd.** The decoupled TPX engine builds computation graphs explicitly, making backpropagation easy to understand and extend.
- **Research ready.** Prototype new layer types, optimizers, custom operators and storage formats with minimal boilerplate.
- **Instantly familiar API.** If you have worked with mainstream deep learning frameworks, the mental model carries over — spend your time on internals, not syntax.

## Installation

### Binaries

```bash
# CPU wheels from PyPI
pip install tensorplay --upgrade

# CUDA wheels from the TensorPlay CUDA index (choose cu124, cu126, or cu130)
# Keep PyPI as an extra index for runtime dependencies.
pip install tensorplay \
  --index-url https://download.tensorplay.cn/whl/cu124/ \
  --extra-index-url https://pypi.org/simple
```

> **Note:** Make sure your Python version matches the wheel tags (e.g. `cp310` for Python 3.10). For CUDA wheels, the driver and runtime must support the CUDA version of the wheel.

### From Source

Building from source gives you a hackable, debuggable install — the recommended setup for contributors and anyone working on kernels.

#### Prerequisites

- Python >= 3.9, < 3.14
- CMake >= 3.18 (< 4.0)
- A C++20-capable compiler (MSVC 2022 on Windows, GCC/Clang on Linux)
- CUDA Toolkit (optional, for GPU support); set `CMAKE_CUDA_ARCHITECTURES` to target specific GPU architectures
- [Ninja](https://ninja-build.org/) (installed automatically with the build dependencies)

#### Get the TensorPlay Source

```bash
git clone https://github.com/bluemoon-o2/TensorPlay.git
cd TensorPlay
```

#### Install Build Dependencies

Isolated PEP 517 builds fetch everything automatically. For faster iterative development, install the toolchain once and build without isolation:

```bash
pip install -r requirements-build.txt
```

#### Install TensorPlay

TensorPlay is built with scikit-build-core through the standard PEP 517 interface declared in `pyproject.toml`:

```bash
# Full build and install (add -v for verbose output)
pip install .

# Editable install for development
pip install -e . --no-build-isolation

# Or produce a wheel without installing
python -m build --wheel
```

On success, `import tensorplay` picks up the compiled `_C` extension from the installed package.

#### Adjusting Build Options (Optional)

Environment variables drive the build — no `-D` flags needed:

```bash
# CPU-only build
USE_CUDA=OFF pip install .

# Target specific GPU architectures
CMAKE_CUDA_ARCHITECTURES="70;75;86" pip install .
```

| Variable | Default | Description |
| ---- | ---- | ---- |
| `USE_CUDA` | auto-detect | Enable/disable the CUDA build |
| `BUILD_TESTS` | `OFF` | Build the C++ test suite |
| `USE_BLAS` / `USE_ONEDNN` | `ON` | BLAS acceleration / oneDNN primitives |
| `MAX_JOBS` | machine default | Cap compile parallelism |
| `DEBUG` | unset | Build with `-O0 -g` |
| `CMAKE_CUDA_ARCHITECTURES` | native | Semicolon-separated GPU architectures |
| `TENSORPLAY_BUILD_VERSION` / `TENSORPLAY_BUILD_NUMBER` | unset | Override the package version (used by release CI) |

All `USE_*`, `BUILD_*` and `CMAKE_*` environment variables are forwarded to CMake automatically — no extra flags required.

## Getting Started

### Automatic Differentiation

```python
import tensorplay as tp

x = tp.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = tp.Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)

z = x.matmul(y) + tp.ones_like(x)
loss = z.sum()
loss.backward()

print(x.grad)  # [[6., 6.], [6., 6.]]
```

### Defining a Neural Network

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
print(model)  # auto-generated architecture visualization
```

### Training Loop

```python
from tensorplay.data import DataLoader, TensorDataset

train_data = TensorDataset(tp.randn(100, 10), tp.randn(100, 1))
train_loader = DataLoader(dataset=train_data, batch_size=8, shuffle=True)

for batch_x, batch_y in train_loader:
    predictions = model(batch_x)
    # ... compute loss, call loss.backward(), step the optimizer
```

Structured tutorials — linear regression from scratch, MNIST CNN classification, custom datasets, model saving/loading with `.mega` + `state_dict` — live at [tensorplay.cn](https://www.tensorplay.cn/en/guide/tutorials).

## Testing

The Python test suite runs with pytest against an installed (or built in-place) package:

```bash
pytest test/
```

CI builds the full wheel matrix (Python 3.9–3.13; CPU on Linux x86_64/aarch64, macOS arm64, and Windows x86_64; CUDA on Linux and Windows x86_64) and validates every wheel on every pull request and push to `main`; see [.github/workflows/](.github/workflows/) (`pull`, `trunk`, `publish`, `lint`). Lint rules live under `[tool.ruff]` in `pyproject.toml`.

## Resources

- **Documentation:** [tensorplay.cn](https://www.tensorplay.cn/)
- **Tutorials:** [tensorplay.cn/en/guide/tutorials](https://www.tensorplay.cn/en/guide/tutorials)
- **Benchmarks:** [benchmark/](benchmark/)
- **Community:** [Discord](https://discord.gg/u6T5e2kGJm)

## Releases and Contributing

The package version lives in [`version.txt`](version.txt) (single source of truth; development installs carry a `+git<sha>` suffix). Releases are cut by pushing a `vMAJOR.MINOR.0` tag; patch tags such as `v1.2.1` do not publish binaries. The CI pipeline builds and validates the full wheel matrix, publishes CPU wheels to PyPI, and uploads CUDA wheels to a GitHub Release before deploying their PEP 503 index to Cloudflare Pages.

The publish workflow uses the `pypi` GitHub environment for PyPI trusted publishing. Python versions and CPU/CUDA platform runners are configured in [`.github/wheel-platforms.json`](.github/wheel-platforms.json); the current CUDA variants are `cu124`, `cu126`, and `cu130` in [`.github/cuda-variants.json`](.github/cuda-variants.json). Adding a future entry such as `{ "variant": "cu132", "toolkit": "13.2.1" }` builds and publishes another CUDA index in the same Pages deployment. CUDA release wheel versions carry a local `+cuXXX` suffix so different CUDA variants can coexist in one GitHub Release. CUDA index deployment requires the `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` repository secrets; the optional `CLOUDFLARE_PAGES_PROJECT_NAME` repository variable defaults to `tensorplay-pypi`.

We welcome contributions of all kinds — bug fixes, documentation, new features. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and coding standards.

<a href="https://github.com/bluemoon-o2/TensorPlay/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=bluemoon-o2/TensorPlay&columns=10" alt="Contributors" />
</a>

## License

TensorPlay is licensed under the [Apache 2.0 License](LICENSE).

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
