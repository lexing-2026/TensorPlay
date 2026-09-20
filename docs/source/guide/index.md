# Getting Started

This guide is the fastest path from "I just installed TensorPlay" to "I'm training a model."
It assumes no prior knowledge of TensorPlay, only basic Python. Each page is short, has a
copy-paste example you can run, and points you to deeper references when you want them.

```{toctree}
:maxdepth: 1

tensors
autograd
models
training
```

## What TensorPlay is

TensorPlay is a tensor library for deep learning. It gives you three things you will use on
every line of code:

- **Tensors** — multi-dimensional arrays that carry a device, a data type, and support
  efficient vectorized math (`tensorplay.Tensor`).
- **Automatic differentiation** — every operation records itself into a graph, so for any
  scalar loss you can call `loss.backward()` and get the gradient of that loss with respect
  to every tensor that was involved (`tensorplay.autograd`).
- **A neural-network toolkit** — layers and models expressed as `tensorplay.nn.Module`,
  optimizers such as `SGD` and `Adam` (`tensorplay.optim`), and data helpers such as
  `DataLoader` and `TensorDataset` (`tensorplay.utils.data`).

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

## A learning path

The four pages above walk you through a single arc, in order. Each one depends on the
previous, so start at the beginning if you are new.

| Page | You will learn | You end up able to |
| --- | --- | --- |
| [Tensors](tensors.md) | creating, inspecting, indexing, and combining tensors | move data in and out of tensors and do vectorized math |
| [Autograd](autograd.md) | `requires_grad`, `backward()`, `.grad`, `no_grad` | compute gradients for the parameters of any model |
| [Models](models.md) | `nn.Module`, layers, parameters, `state_dict` | define and inspect a neural network |
| [Training](training.md) | loss, optimizer, `DataLoader`, the training loop | train a real classifier and save the trained model |

## Where to go next

- **API reference** — every function and class, grouped by area. Start at
  [tensorplay](../tensorplay.md) for the core tensor API, then [nn](../nn.md) for layers and
  losses, and [optim](../optim.md) for optimizers.
- **Deep-dive notes** — the [notes](../notes/index.md) section explains the concepts behind
  the code: how autograd builds its graph, broadcasting rules, serialization, randomness,
  and more.
