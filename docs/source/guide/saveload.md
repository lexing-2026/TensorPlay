# Save and Load the Model

After training, you want to keep the weights so you can reuse them later, serve them, or
resume training from a checkpoint. TensorPlay saves and loads models through their
`state_dict` — the mapping of parameter names to tensors.

```python
import tensorplay as tp
from tensorplay import nn

model = nn.Linear(4, 2)
tp.save(model.state_dict(), 'model.mega')
model.load_state_dict(tp.load('model.mega'))
```

## What `state_dict` is

Every `nn.Module` exposes `state_dict()`, an ordered mapping from parameter name to tensor.
For a `Linear(4, 2)` it looks like:

```python
model = tp.nn.Linear(4, 2)
print(model.state_dict().keys())   # ['weight', 'bias']
```

This is the canonical form of your model: the architecture is the class, and the `state_dict`
is the learned values. A `.mega` file stores tensors and JSON primitives, so it is built to
hold `state_dict`s rather than whole module objects — you always rebuild the architecture and
load the weights into it.

## What the `.mega` file holds

`tp.save` writes a `.mega` file that can store tensors, and plain Python values (numbers,
strings, lists, dicts) nested inside them. `tp.load` restores it. This is exactly what a
`state_dict` (a mapping of names to tensors) or a checkpoint dict is, so both round-trip
cleanly.

Attempting to save the module object itself is not supported — `tp.save(model, ...)` raises a
`TypeError` informing you that `.mega` holds tensors and JSON primitives. To move a model to a
new process, save its `state_dict` and `load_state_dict` it into a fresh instance of the same
architecture.

## Save, rebuild, load

The standard pattern with a trained model:

```python
import tensorplay as tp
from tensorplay import nn

model = nn.Linear(4, 2)          # imagine this is your trained model
tp.save(model.state_dict(), 'model.mega')

new_model = nn.Linear(4, 2)      # same architecture
new_model.load_state_dict(tp.load('model.mega'))
```

`load_state_dict` copies the stored weights in by name and returns an `_IncompatibleKeys`
object listing any keys that were missing or unexpected — a mismatch usually means you changed
the architecture.

## Checkpointing during training

For training, save more than the weights: the optimizer state and the epoch let you resume
exactly where you stopped.

```python
import tensorplay as tp
from tensorplay import nn, optim

model = nn.Linear(4, 2)
opt = optim.SGD(model.parameters(), lr=0.01)
epoch = 5

# save a checkpoint (a plain dict of tensors and numbers)
tp.save({
    'model': model.state_dict(),
    'optimizer': opt.state_dict(),
    'epoch': epoch,
}, 'checkpoint.mega')

# resume
checkpoint = tp.load('checkpoint.mega')
model.load_state_dict(checkpoint['model'])
opt.load_state_dict(checkpoint['optimizer'])
start_epoch = checkpoint['epoch'] + 1
```

## Common pitfalls

- **Call `load_state_dict` on a fresh model of the same architecture**, not on the already
  trained object, unless that is what you intend.
- **Keep the receiving model's architecture identical.** When the architecture changes, the
  parameter keys change and `load_state_dict` reports the mismatch rather than misloading.
- **Move tensors to the right device after loading** if you train on a GPU and load on CPU or
  vice versa.

## Where to go next

- The [checkpoint page](../checkpoint.md) covers resuming training, distributed checkpoints,
  and the full serialization API.
- The [serialization note](../notes/serialization.md) explains the file format and the rules
  around what can be loaded.