# Training

Training a model is the same small loop every time: run the model forward, measure how wrong
it is with a loss, backpropagate to get gradients, and nudge the parameters with an optimizer.
This page wires together the [tensors](tensors.md), [autograd](autograd.md), and
[models](models.md) you have already seen.

We will build a complete training script piece by piece. Every block below runs as-is in
order, so you can follow along in a Python file or a REPL.

## The pieces

First, the loss and the optimizer. A loss measures error, an optimizer updates parameters:

```python
import tensorplay as tp
from tensorplay import nn, optim

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)
loss_fn = nn.BCEWithLogitsLoss()          # binary classification, applied to raw logits
opt = optim.Adam(model.parameters(), lr=0.02)
```

`optim.SGD(model.parameters(), lr=0.01)` is the other optimizer you will reach for. Both take
the parameters from `model.parameters()`, which is exactly why you build models as
`nn.Module` subclasses — the module collects them for you.

## The data

`tensorplay.utils.data` provides `DataLoader`, `TensorDataset`, and `Dataset`. Build a
`TensorDataset` from feature and label tensors, then wrap it in a `DataLoader`, which yields
batches:

```python
from tensorplay.utils.data import DataLoader, TensorDataset

features = tp.randn(200, 4)
labels = (features[:, 0] + features[:, 1] > 0).to(tp.int64).reshape(-1, 1)
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
```

Here `labels` is an integer `(200, 1)` column of `0`s and `1`s, built from a comparison. The
`.to(tp.int64)` converts the boolean comparison result to a numeric type the loss accepts.

## The loop

Now the fixed rhythm. Before each batch, clear the gradients with `opt.zero_grad()`. Then
compute the loss, call `loss.backward()` to fill in gradients, and `opt.step()` to apply
them:

```python
for epoch in range(8):
    total = 0.0
    for batch_x, batch_y in loader:
        opt.zero_grad()
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch {epoch}: loss {total / len(loader):.4f}")
```

That inner block is the entire training step. Everything else is bookkeeping: shuffling the
loader, moving data to a device, logging the loss, and evaluating.

## Evaluate

When you are done training, run a forward pass only and compare predictions to the labels. The
evaluation is wrapped in `tp.no_grad()` so it is faster and does not build a graph:

```python
with tp.no_grad():
    correct = (model(features) > 0).to(tp.int64).eq(labels).sum().item()
print(f"accuracy: {correct / len(features):.2f}")
```

`BCEWithLogitsLoss` works on the raw output of the model (no activation), which is numerically
stable. Add `nn.Sigmoid` after the last layer only when you want a probability, not when you
are computing this loss.

## Move to a device

Inputs, targets, and the model should all live on the same device. `.to(device)` moves a
tensor, and `model.to(device)` moves all of a model's parameters:

```python
device = tp.device('cpu')        # use a GPU device here when available
model = model.to(device)

for batch_x, batch_y in loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    out = model(batch_x)         # model parameters are already on device
```

See the [CUDA page](../cuda.md) for how to pick and use an accelerator.

## Check the gradient flow

A training loop that never improves usually fails before the optimizer. The fastest diagnosis
is to confirm gradients actually reached the parameters:

```python
out = model(features[:8])
loss = loss_fn(out, labels[:8])
loss.backward()
for name, param in model.named_parameters():
    print(name, param.grad is not None)
```

If `param.grad` is `None`, the parameter is not reachable from the loss. The usual causes are
a parameter that was never used in `forward`, a `no_grad` block that wraps too much, or a
`.detach()` that cut the graph.

## Putting it all together

Here is the whole script, ready to run:

```python
import tensorplay as tp
from tensorplay import nn, optim
from tensorplay.utils.data import DataLoader, TensorDataset

features = tp.randn(200, 4)
labels = (features[:, 0] + features[:, 1] > 0).to(tp.int64).reshape(-1, 1)
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
)

loss_fn = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=0.02)

for epoch in range(8):
    total = 0.0
    for batch_x, batch_y in loader:
        opt.zero_grad()
        out = model(batch_x)
        loss = loss_fn(out, batch_y)
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"epoch {epoch}: loss {total / len(loader):.4f}")

with tp.no_grad():
    correct = (model(features) > 0).to(tp.int64).eq(labels).sum().item()
print(f"accuracy: {correct / len(features):.2f}")
```

## Where to go next

- The [optim API reference](../optim.md) lists every optimizer and its options.
- The [data documentation](../data.md) covers datasets, loaders, and sampling in depth.
- Once you can train a model, the natural next steps are saving the best weights
  ([models](models.md)), automatic mixed precision ([amp](../amp.md)), and running on a GPU
  ([cuda](../cuda.md)).
