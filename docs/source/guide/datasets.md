# Datasets and DataLoaders

In the previous pages you fed a model small tensors by hand. Real training uses far more data
than fits in memory, so you need a way to (1) describe a dataset, and (2) stream it in small
batches. That is exactly what `tensorplay.utils.data` gives you.

```python
import tensorplay as tp
from tensorplay.utils.data import Dataset, TensorDataset, DataLoader
```

## A dataset is just something you can index

The `Dataset` base class has two methods. `__len__` says how many samples there are, and
`__getitem__` returns the sample at index `i`. Any object with these two methods can be
wrapped in a `DataLoader`:

```python
class Numbers(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return i

ds = Numbers(10)
print(len(ds), ds[3])   # 10 3
```

For machine learning, a sample is usually a pair — features and a label. `TensorDataset`
wraps one or more tensors into a dataset where index `i` returns the `i`-th row of each:

```python
features = tp.randn(100, 4)
labels = (features[:, 0] > 0).to(tp.int64).reshape(-1, 1)
dataset = TensorDataset(features, labels)
x, y = dataset[0]
print(x.shape, y.shape)   # (4,) (1,)
```

## A DataLoader batches it

`DataLoader` turns a dataset into an iterable that yields batches. The two arguments you will
use constantly are `batch_size` (how many samples per batch) and `shuffle` (whether to randomize
the order each epoch):

```python
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)   # (16, 4) (16, 1)
    break
```

By default the loader is iterable-style: you loop over it to get batches. Other options you may
need:

- `drop_last=True` — drop the final incomplete batch when the dataset size is not a multiple of
  `batch_size`.
- `num_workers=4` — load several batches in parallel processes. The default of `0` loads in the
  main process, which is fine for small data.
- `pin_memory=True` — useful when you are training on a GPU; ask your data loader to put batches
  into pinned memory for faster transfer.

## Choosing which samples to use: samplers

`shuffle=True` randomizes order, but for anything more specific — a fixed random subset,
weighted draws for imbalanced classes — pass a *sampler* instead. A sampler is an iterable of
indices; the loader draws batches from it:

```python
from tensorplay.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

data = TensorDataset(tp.arange(6))

# draw 4 indices, with the first two classes excluded entirely (weight 0)
weights = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
sampler = WeightedRandomSampler(weights, num_samples=4, replacement=True)
loader = DataLoader(data, sampler=sampler)
for (batch,) in loader:
    print(batch)      # single-sample batches drawn from indices 2..5
```

The commonly used ones:

- `RandomSampler(dataset)` — what `shuffle=True` does internally.
- `SequentialSampler(dataset)` — indices `0..n-1` in order.
- `SubsetRandomSampler(indices)` — shuffle within a fixed subset; the classic
  train/validation split.
- `WeightedRandomSampler(weights, num_samples)` — oversample rare classes.
- `DistributedSampler(dataset)` — one shard per process; see the [DDP note](../notes/ddp.md).

`sampler` and `shuffle` are mutually exclusive — passing both raises a `ValueError`. If you
need full control over batch *composition*, `batch_sampler` takes an iterable of index lists,
one per batch.

## Building datasets out of datasets

Four helpers turn existing datasets into new ones without copying data:

```python
from tensorplay.utils.data import ConcatDataset, StackDataset, Subset, TensorDataset

ds = TensorDataset(tp.arange(10), tp.arange(10) * 10)

sub = Subset(ds, [3, 1, 4])       # a view of three chosen indices
print(sub[0])                     # the (3, 30) sample

merged = ConcatDataset([ds, ds])  # end-to-end, length 20
print(len(merged))

paired = StackDataset(tp.arange(5), tp.arange(5) * 2)   # zip two equal-length tensors
print(paired[2])                  # the (2, 4) sample
```

`Subset` is also how you carve a validation set out of one dataset:

```python
from tensorplay.utils.data import Subset

n = len(ds)
train_ds = Subset(ds, range(0, int(n * 0.8)))
val_ds = Subset(ds, range(int(n * 0.8), n))
```

## Streaming data: `IterableDataset`

When the data does not have a known length — a log file that keeps growing, a message queue,
a generator — implement `IterableDataset` and define `__iter__` instead of `__len__` and
`__getitem__`:

```python
from tensorplay.utils.data import DataLoader, IterableDataset

class Countdown(IterableDataset):
    def __iter__(self):
        for i in range(10):
            yield i

loader = DataLoader(Countdown(), batch_size=4)
print([batch.tolist() for batch in loader])
# [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
```

Iterable-style loaders cannot `shuffle` (there are no indices to permute); if you need
randomization, shuffle inside `__iter__` yourself. Multi-worker loading needs care: each
worker gets its own copy of the dataset and iterates *the whole stream*, so with
`num_workers=2` every sample would arrive twice. Shard the stream by worker instead:

```python
from tensorplay.utils.data import DataLoader, IterableDataset, get_worker_info

class Sharded(IterableDataset):
    def __iter__(self):
        info = get_worker_info()
        start = 0 if info is None else info.id
        step = 1 if info is None else info.num_workers
        for i in range(start, 10, step):     # worker 0: 0,2,4,...; worker 1: 1,3,5,...
            yield i

loader = DataLoader(Sharded(), batch_size=4, num_workers=2)
print(sorted(x for batch in loader for x in batch.tolist()))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] — every element exactly once
```

`get_worker_info()` returns `None` in the main process, which is why the single-process path
falls back to a step of 1.

## Custom collation

Collation is the step that turns a list of samples into one batch. The default collate stacks
tensors and passes numbers through, which covers most cases. When your samples do not fit
that shape — variable-length sequences, nested dicts, PIL images — pass `collate_fn`:

```python
from tensorplay.utils.data import DataLoader, Dataset

class Squares(Dataset):
    def __len__(self):
        return 8
    def __getitem__(self, i):
        return i

loader = DataLoader(Squares(), batch_size=4, collate_fn=lambda batch: tp.tensor(batch) * 10)
print([b.tolist() for b in loader])
# [[0, 10, 20, 30], [40, 50, 60, 70]]
```

A collate function receives the *list* of samples for one batch and returns anything the
training loop can consume — a tensor, a dict of tensors, a tuple of padded sequences.

## Loading images

For image datasets, `tensorplay.vision.datasets` provides ready-made datasets such as `MNIST`
and `CIFAR10`. Each returns a `(image, label)` pair, where image is a `PIL` image or a tensor:

```python
from tensorplay import vision

train_data = vision.datasets.MNIST(
    root='./data', train=True, download=True)
```

The `root` is where the dataset is stored on disk and `download=True` fetches it the first time.
You will usually wrap this in `DataLoader` and apply transforms — see
[Transforms](transforms.md).

## An end-to-end data pipeline

Here is the typical shape: build a dataset, apply a transform (for images, convert them to
tensors), and wrap it in a loader:

```python
from tensorplay.utils.data import TensorDataset, DataLoader

features = tp.randn(64, 10)
labels = tp.randint(0, 2, (64, 1))          # (64, 1) integer labels
dataset = TensorDataset(features, labels)
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)     # (8, 10) (8, 1)
    break
```

`tp.randint(0, 2, (64, 1))` makes random integer labels; use `TensorDataset` when your data is
already tensors, and a `Dataset` subclass when it is anything else (files, cloud objects, ...).

## Where to go next

- The [data documentation](../data.md) covers every option of `DataLoader`, samplers, collating,
  and multi-process loading in detail.
- [Transforms](transforms.md) next, for preprocessing images before you feed them to a model.
