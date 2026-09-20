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
