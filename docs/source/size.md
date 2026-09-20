# Size

`tensorplay.Size` is the lightweight object that holds a tensor's shape. It behaves like a
tuple of integers.

```python
import tensorplay as tp

x = tp.zeros(2, 3, 4)
print(x.shape)     # tensorplay.Size([2, 3, 4])
print(x.size())    # same
print(len(x.shape))  # 3
```

`size()` on a tensor accepts an optional dimension to return a single axis length:

```python
print(x.size(0))    # 2
print(x.size(-1))   # 4
```

A `Size` is created automatically by the shape of a tensor, and you can build one directly:

```python
s = tp.Size([3, 4])
print(s)           # tensorplay.Size([3, 4])
print(tuple(s))    # (3, 4)
```

Because it is a tuple-like sequence, a `Size` can be unpacked or sliced, and is what `reshape`
expects as its target shape.

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.Size
```

## Where to go next

- [Tensor attributes](tensor_attributes.md) — the rest of a tensor's metadata.
