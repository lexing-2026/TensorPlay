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

## What `state_dict` contains

Parameters, plus every *persistent* buffer. A buffer is module state that is tracked and
moved with the model but not learned — BatchNorm's running statistics are the standard
example. Buffers registered with `persistent=False` are deliberately excluded:

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.register_buffer("running", tp.zeros(3))                    # saved
        self.register_buffer("scratch", tp.zeros(3), persistent=False)  # not saved

net = Net()
print(list(net.state_dict().keys()))
# ['running', 'fc.weight', 'fc.bias']
```

So a checkpoint restores everything the architecture itself does not reconstruct: learned
weights *and* accumulated statistics. See [Models](models.md) for buffers in general.

## Loading part of a model

`load_state_dict` is strict by default: the keys must match exactly, and a mismatch raises
with a report of what is missing and what is unexpected. `strict=False` relaxes that, which
is exactly what transfer learning needs — keep the pretrained backbone's weights, let a new
head keep its fresh initialization:

```python
backbone = nn.Sequential(nn.Linear(784, 128), nn.ReLU())
head = nn.Linear(128, 10)
model = nn.Sequential(backbone, head)

# a checkpoint holding only the backbone's weights
result = backbone.load_state_dict(backbone.state_dict(), strict=False)
print(result.missing_keys, result.unexpected_keys)   # [] []

# going the other way: keys in the file that no module wants are skipped
dst = nn.Linear(4, 2)
src_sd = dict(nn.Linear(4, 2).state_dict())
src_sd["extra"] = tp.zeros(1)
result = dst.load_state_dict(src_sd, strict=False)
print(result.unexpected_keys)   # ['extra']
print(result.missing_keys)      # [] — everything else loaded
```

`load_state_dict` never raises on mismatch under `strict=False`; it returns an
`_IncompatibleKeys` object so you can inspect `missing_keys` and `unexpected_keys` yourself.
If keys went missing that you *expected* to load, the architecture does not line up with the
checkpoint — fix that first, because silently re-initialized layers train from scratch.

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