# DLPack

[DLPack](https://dmlc.github.io/dlpack/) is a neutral, zero-copy exchange format for tensors.
It lets two different libraries hand a tensor back and forth without copying the data,
describing only the shape, strides, data type, and device. TensorPlay interoperates with it
through `tensorplay.from_dlpack`.

```python
import tensorplay as tp

# a DLPack capsule is an opaque handle to a tensor
x = tp.ones(3, 4)
capsule = tp.to_dlpack(x)
y = tp.from_dlpack(capsule)   # wraps the same data, no copy
print(y.shape)                # (3, 4)
```

`tensorplay.from_dlpack` wraps data owned by another library, and `tensorplay.to_dlpack`
hands ours out in the same format. Because the data is shared rather than copied, both ends
must respect the tensor's lifetime and the device contract — the capsule owns the underlying
memory and the two sides agree on when it is valid.

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.from_dlpack
    tensorplay.to_dlpack
```

## Where to go next

- [Tensors](tensorplay.md) — the tensor APIs `from_dlpack` and `to_dlpack` interoperate with.
- [NumPy interop](notes/serialization.md) — the other zero-copy conversion path.
