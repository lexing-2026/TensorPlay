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
