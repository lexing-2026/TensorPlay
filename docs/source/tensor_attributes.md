# Tensor attributes

A tensor carries metadata alongside its data: a data type, a device, and a shape. This page
covers the two that describe *what* an element is and *where* it lives. Shape is described in
[Size](size.md).

## Data type

A tensor's data type (dtype) states how each element is stored and what range of values it can
hold. The common dtypes are exposed as constants:

```python
import tensorplay as tp

print(tp.float32)   # the 32-bit floating-point dtype
print(tp.int64)     # the 64-bit integer dtype
print(tp.bool)      # a boolean dtype
```

Pass a dtype to a creation function or `Tensor` to control it:

```python
x = tp.tensor([1, 2, 3], dtype=tp.float32)
print(x.dtype)      # tensorplay.float32
```

There are integer, floating-point, and complex dtypes (`tensorplay.complex64`,
`tensorplay.complex128`). Operations that mix dtypes promote according to TensorPlay's
type-promotion rules, so a `float32` tensor times an `int64` tensor yields a `float32` tensor.

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.dtype
    tensorplay.get_default_dtype
    tensorplay.set_default_dtype
```

## Device

A tensor lives on a device — `cpu`, or a GPU such as `cuda`. The device determines where the
data is physically stored and which kernels can run on it:

```python
x = tp.tensor([1.0, 2.0])
print(x.device)      # cpu
```

Use `.to(device)` to move a tensor (and `model.to(device)` to move a model) to another device.
See the [CUDA](cuda.md) page for working with accelerators.

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.device
    tensorplay.get_default_device
    tensorplay.set_default_device
```

## Defaults

`set_default_dtype` and `set_default_device` change the dtype and device used for tensors
created without an explicit one, and `get_default_dtype` / `get_default_device` read them back.

## Where to go next

- [Tensors](tensorplay.md) — the full tensor creation and math API.
- [Type information](type_info.md) — `finfo` and `iinfo`, the per-dtype limits.
