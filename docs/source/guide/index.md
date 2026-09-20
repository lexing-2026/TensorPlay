# Getting Started

This guide is the fastest path from "I just installed TensorPlay" to "I trained a model."
It is organized as a small series of lessons, each short enough to read and run in one sitting.
Every example is copy-paste runnable against an installed TensorPlay.

```{toctree}
:maxdepth: 1

tensors
datasets
transforms
models
autograd
training
saveload
```

## The path

Follow the pages in order. Each one builds on the previous.

1. [Tensors](tensors.md) — create, inspect, index, and do math with tensors.
2. [Datasets and DataLoaders](datasets.md) — organize data and stream it in batches.
3. [Transforms](transforms.md) — preprocess and augment data before it reaches a model.
4. [Models](models.md) — build a neural network with `nn.Module`.
5. [Autograd](autograd.md) — make tensors learnable and get gradients automatically.
6. [Training](training.md) — the loss + optimizer + loop that makes the model learn.
7. [Save and Load](saveload.md) — keep the trained weights and checkpoints.

After those seven lessons you can train, evaluate, and persist a real model.

## What TensorPlay is

TensorPlay is a tensor library for deep learning. It gives you three things you will use on
every line of code:

- **Tensors** — multi-dimensional arrays that carry a device, a data type, and support
  efficient vectorized math (`tensorplay.Tensor`).
- **Automatic differentiation** — every operation records itself into a graph, so for any
  scalar loss you can call `loss.backward()` and get the gradient of that loss with respect
  to every tensor that was involved (`tensorplay.autograd`).
- **A neural-network toolkit** — layers and models expressed as `tensorplay.nn.Module`,
  optimizers such as `SGD` and `Adam` (`tensorplay.optim`), vision transforms
  (`tensorplay.vision.transforms`), and data helpers such as `DataLoader` and
  `TensorDataset` (`tensorplay.utils.data`).

A defining design choice: the engine is explicit and readable. The graph that `backward()`
walks is real code you can step through, which makes TensorPlay a good place to *learn* how
the machinery works, not just to use it.

## Install and check it works

You can install the latest release, or build from source. See the [README](https://github.com/lexing-2026/TensorPlay/blob/main/README.md)
for the full install options including GPU variants. Once installed, run this smoke test:

```python
import tensorplay as tp

x = tp.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

If that prints `2 4 6` as a gradient, your installation is working.

## Where to go next

- **API reference** — every function and class, grouped by area. Start at
  [tensorplay](../tensorplay.md) for the core tensor API, then [nn](../nn.md) for layers and
  losses, and [optim](../optim.md) for optimizers.
- **Deep-dive notes** — the [notes](../notes/index.md) section explains the concepts behind
  the code: how autograd builds its graph, broadcasting rules, serialization, randomness,
  and more.