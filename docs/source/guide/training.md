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

## A validation split

Accuracy on training data flatters the model. Carve out a slice of the data, evaluate on it
each epoch with the model switched to eval mode, and you will see whether the model is
learning or memorizing:

```python
from tensorplay.utils.data import DataLoader, Subset, TensorDataset

n_train = int(len(dataset) * 0.8)
train_ds = Subset(dataset, range(0, n_train))
val_ds = Subset(dataset, range(n_train, len(dataset)))
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

def evaluate(model, loader):
    model.eval()                       # freeze dropout / batch statistics
    correct = 0
    with tp.no_grad():                 # no graph, no gradients
        for batch_x, batch_y in loader:
            pred = (model(batch_x) > 0).to(tp.int64)
            correct += pred.eq(batch_y).sum().item()
    model.train()                      # back to training mode
    return correct / len(loader.dataset)

print(f"val accuracy: {evaluate(model, val_loader):.2f}")
```

The two mode switches are the part people forget: `model.eval()` before measuring,
`model.train()` before the next epoch. The `Subset`/`DataLoader` split shown here is the
whole recipe — no separate framework machinery needed.

## Learning-rate schedules

Almost every optimizer benefits from a learning rate that changes over training — large
early, small late. `tensorplay.optim.lr_scheduler` holds the standard shapes. A scheduler
wraps an optimizer and mutates its learning rate each time you step it:

```python
import tensorplay.optim as optim

opt = optim.Adam(model.parameters(), lr=0.02)
scheduler = optim.lr_scheduler.StepLR(opt, step_size=2, gamma=0.5)

for epoch in range(6):
    print(epoch, opt.param_groups[0]['lr'])   # what this epoch will use
    for batch_x, batch_y in train_loader:
        opt.zero_grad()
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        opt.step()
    scheduler.step()    # after the epoch's optimizer steps
```

The learning rates across the six epochs are `0.02, 0.02, 0.01, 0.01, 0.005, 0.005` —
halved every two epochs. The rule for ordering: call `scheduler.step()` *after*
`opt.step()`, once per epoch for epoch-schedulers like these. Also available:
`MultiStepLR` (drop at named epochs), `ExponentialLR` (decay every step), `CosineAnnealingLR`
(smooth cosine decay, a modern default), `OneCycleLR` (warm up then anneal, stepped per
*batch*), `LambdaLR` (your own function), `ReduceLROnPlateau` (drop when a metric stops
improving — the one scheduler that takes the metric, `scheduler.step(val_loss)`), and
combinators `SequentialLR`/`ChainedScheduler`.

## Gradient clipping

Recurrent models and unstable configurations can produce a batch with an enormous gradient,
and one wild step can throw the weights somewhere the optimizer never recovers from.
`clip_grad_norm_` rescales the gradient vector to a maximum length, *between* `backward()`
and `opt.step()`:

```python
from tensorplay.nn.utils import clip_grad_norm_

x = tp.tensor([3.0, 4.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
total = clip_grad_norm_(x, max_norm=1.0)
print(total.item())    # 10.0 — the norm before clipping
print(x.grad)          # [0.6, 0.8] — rescaled to length 1
```

The returned value is the pre-clip total norm, which is worth logging: a steadily growing
norm is an early warning of divergence. `clip_grad_value_` is the cruder sibling that cuts
each element into a fixed range instead of scaling the whole vector.

A training loop with both additions looks like:

```python
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
opt.step()
scheduler.step()
```

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
