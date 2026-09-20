# Tensor views

A view is a tensor that shares its underlying data with another tensor. Reshaping, selecting
dimensions, or reversing a tensor are all *cheap* — instead of copying the elements, the view
stores different metadata (shape, strides) over the same storage. This is why operations like
`view`, `transpose`, and `squeeze` are instant and memory-efficient.

```python
import tensorplay as tp

x = tp.arange(12).reshape(3, 4)
y = x.t()          # a transposed view
print(x.data_ptr() == y.data_ptr())  # True — same underlying data
y[0, 0] = 99
print(x[0, 0].item())   # 99 — the change is visible through the view
```

The rule of thumb: **an operation that only changes shape, strides, or layout produces a
view; an operation that creates new elements copies.** `reshape` may return a view or a copy
depending on the memory layout — use `view` when you know the tensor is contiguous and want to
guarantee no copy.

## The view functions

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.as_strided
    tensorplay.narrow
    tensorplay.expand
    tensorplay.squeeze
    tensorplay.unsqueeze
    tensorplay.select
    tensorplay.t
    tensorplay.transpose
    tensorplay.permute
    tensorplay.view
    tensorplay.reshape
    tensorplay.flatten
    tensorplay.split
    tensorplay.chunk
```

Common ones in context:

```python
x = tp.arange(6).reshape(2, 3)
print(x.transpose(0, 1).shape)   # (3, 2)
print(x.view(6))                 # flatten, no copy
print(x.narrow(1, 0, 2).shape)   # (2, 2) — rows 0:2 of the second dim
b = tp.ones(1, 3)
print(b.expand(4, 3).shape)      # (4, 3) — broadcast a size-1 dim, no copy
```

## When data is not shared

Some operations rebuild the data and do *not* share storage:

- `clone()` always copies.
- `contiguous()` returns a copy when the tensor is non-contiguous (e.g. after `transpose`).
- `reshape` copies when the required shape is not reachable with the current strides.
- `detach()` shares data but cuts the autograd graph (see [Autograd](guide/autograd.md)).

## Where to go next

- [Tensors](tensorplay.md) — the full tensor API.
- [Serialization](notes/serialization.md) — how views interact with saving and loading.