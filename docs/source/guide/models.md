# Models

A neural network is a stack of parameterized operations. TensorPlay expresses one as a
subclass of `tensorplay.nn.Module`: you override `forward` to define how inputs become
outputs, and TensorPlay automatically collects every tensor you assign as an attribute into
the module's list of parameters.

```python
import tensorplay as tp
from tensorplay import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

model = MLP(4, 8, 1)
```

Calling a module runs its `forward` method:

```python
x = tp.randn(3, 4)
print(model(x).shape)  # (3, 1)
```

## Why subclass `Module`

The two things a module must do for you are (1) hold parameters and (2) run `forward`.
TensorPlay's `Module` provides a lot on top:

- **The module is callable** — `model(x)` calls `forward`.
- **It tracks its parameters** — parameters registered as attributes (or added with
  `register_parameter`) are discoverable, so you never have to keep a list yourself.
- **It tracks its submodules** — a nested module like `self.fc1` is found automatically.
- **It has a `state_dict`** — a mapping of parameter names to tensors, which is what you
  save and load.

```python
for name, param in model.named_parameters():
    print(name, param.shape)
```

## Layers you will use

The `tensorplay.nn` namespace has the standard building blocks. A few common ones:

- `nn.Linear(in, out)` — a fully-connected layer.
- `nn.Conv2d(in, out, kernel)` — a convolutional layer for images.
- `nn.ReLU`, `nn.Sigmoid`, `nn.Softmax` — activation functions.
- `nn.Flatten` — reshape a multi-dimensional input into a single dimension.
- `nn.Dropout(p)` — regularization for training.
- `nn.Sequential(*layers)` — stack layers in a single module.

```python
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
```

## Inspecting a model

`print(model)` shows the structure and the shape of each parameter; TensorPlay renders an
architecture visualization automatically:

```python
print(model)
```

Use `model.parameters()` or `model.named_parameters()` to get the learnable tensors, which is
what you hand to an optimizer. `model.children()` iterates over submodules, and
`model.named_modules()` over the whole tree.

## Parameters are just tensors

A parameter is a tensor that has `requires_grad=True` and lives on the model. That means
everything from the [Autograd](autograd.md) page applies to it directly: run
`loss.backward()`, and each parameter's gradient lands in its `.grad`.

## Buffers: state without gradients

Some model state should be tracked by the module but is *not* learned — BatchNorm's running
statistics are the classic example. That is what a buffer is. Register one with
`register_buffer` and it is carried along by `.to(device)`, included in `state_dict()`, and
excluded from `model.parameters()`:

```python
import tensorplay as tp
from tensorplay import nn

class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("center", tp.zeros(3))            # saved in state_dict
        self.register_buffer("scratch", tp.zeros(3), persistent=False)

s = Scale()
print(list(s.state_dict().keys()))     # ['center'] — non-persistent buffers are not saved
print([name for name, _ in s.named_buffers()])   # ['center', 'scratch']
```

Pass `persistent=False` for throwaway state that should not be serialized — a step counter,
a cached mask. The mechanics of buffers, parameters, and submodule registration are covered
in the [modules note](../notes/modules.md).

## `train()` and `eval()`

Modules carry a training flag, flipped by `model.train()` and `model.eval()`. Two families of
layers read it:

- **Dropout** zeroes elements with probability `p` during training and rescales the survivors
  so the expected activation is unchanged; in eval mode it is an exact pass-through.
- **BatchNorm** updates its running statistics from each training batch and normalizes with
  batch statistics; in eval mode it normalizes with the frozen running statistics instead.

```python
drop = nn.Dropout(p=0.5)
drop.train()
print(drop(tp.ones(10000)).mean().item())   # ≈ 1.0 — half zeroed, survivors scaled by 2
drop.eval()
print(drop(tp.ones(10000)).mean().item())   # exactly 1.0 — pass-through
```

Forgetting to call `model.eval()` before validating is the single most common cause of
"my validation loss looks like training loss": dropout noise and fresh batch statistics leak
into the measurement. Switch with `model.train()` when the next epoch starts. The flag is
recursive — calling it on the top module flips every submodule.

## Initializing weights

`nn.Linear` and the convolution layers initialize themselves with a sensible scheme
(a Kaiming-style uniform for weights, a small symmetric uniform for biases), so a fresh
model is trainable as-is. When you need to take control, `tensorplay.nn.init` provides the
standard fillers, all taking a tensor and modifying it in place:

```python
import tensorplay.nn.init as init

lin = nn.Linear(64, 32)
init.kaiming_uniform_(lin.weight, a=5 ** 0.5)   # the exact scheme Linear defaults to
init.zeros_(lin.bias)

small = nn.Linear(8, 8)
init.constant_(small.weight, 0.01)
```

The full set includes `normal_`, `uniform_`, `trunc_normal_`, `xavier_uniform_`,
`xavier_normal_`, `orthogonal_`, `sparse_`, `eye_`, `dirac_`, and the constant fills. To
(re-)initialize an existing tree of modules, use `apply`, which calls your function on every
submodule:

```python
def reinit(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight)
        init.zeros_(m.bias)

model.apply(reinit)     # visits every submodule once
```

## Freezing parameters

Setting `requires_grad_(False)` on a parameter removes it from the graph: no gradient is
computed for it and the optimizer leaves it alone. This is the transfer-learning move —
keep a pretrained backbone fixed and train only the new head:

```python
backbone = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 128))
head = nn.Linear(128, 10)

for p in backbone.parameters():
    p.requires_grad_(False)

model = nn.Sequential(backbone, head)
opt = tp.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],   # only trainable ones
    lr=1e-3,
)
```

Two details worth knowing:

- Filter by `p.requires_grad` when building the optimizer, or pass `model.parameters()` and
  let the optimizer skip updates whose `.grad` is `None`.
- If *every* tensor in the computation is frozen, `backward()` has nothing to differentiate
  and raises — at least one input to the loss must require grad.

## Save and load

A module's `state_dict` is what you persist. Save it, then rebuild the module and load it
back:

```python
model = MLP(4, 8, 1)

# save
tp.save(model.state_dict(), 'model.mega')

# load into a fresh model
new_model = MLP(4, 8, 1)
new_model.load_state_dict(tp.load('model.mega'))
```

The `.mega` file holds just the parameters, which makes it small and portable. To move weights
from one model to another, always build the receiving model with the same architecture first,
then call `load_state_dict`. See the [serialization note](../notes/serialization.md) for
saving the whole model, other formats, and the rules around what can be loaded.

## Where to go next

- [Training](training.md) — combine a model with a loss and an optimizer to make it learn.
- The [modules note](../notes/modules.md) is a thorough tour of every `Module` capability.
- The [nn API reference](../nn.md) lists every layer and loss available.
