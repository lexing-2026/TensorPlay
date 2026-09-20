# Tensors

A tensor is a multi-dimensional array of numbers. Every value in TensorPlay — inputs, model
parameters, intermediate activations, gradients — is a tensor, so this is the first thing to
learn.

This page covers the four things you do with tensors constantly: **create** one, **inspect**
it, **index** into it, and **do math** with it.

## Creating tensors

The simplest way is `tensorplay.tensor`, which wraps a list, a tuple, or a NumPy array:

```python
import tensorplay as tp

a = tp.tensor([1, 2, 3])                    # a 1-D integer tensor
b = tp.tensor([[1.0, 2.0], [3.0, 4.0]])     # a 2-D floating-point tensor
c = tp.tensor([1, 2, 3], dtype=tp.float32)  # explicit data type
```

You can also create tensors with a shape and no data, or filled with a pattern:

```python
ones = tp.ones(2, 3)          # all ones, shape (2, 3)
zeros = tp.zeros(4)           # all zeros, shape (4,)
eye = tp.eye(3)               # identity matrix, shape (3, 3)
seq = tp.arange(6)            # [0, 1, 2, 3, 4, 5]
grid = tp.linspace(0.0, 1.0, 5)  # five evenly spaced values from 0.0 to 1.0
rand = tp.randn(2, 2)         # random values from a standard normal
```

Two companions are worth knowing early:

- `tensorplay.as_tensor(x)` converts `x` into a tensor but reuses the input's data when
  possible instead of always copying. It is the natural choice when you already have a
  tensor or a NumPy array and want to make sure it is a tensor.
- `tensorplay.from_numpy(arr)` turns a NumPy array into a tensor. `.numpy()` does the reverse.

## Inspecting a tensor

Every tensor exposes its shape, data type, and the device it lives on:

```python
t = tp.tensor([[1.0, 2.0], [3.0, 4.0]])
print(t.shape)    # tensorplay.Size([2, 2])
print(t.dtype)    # tensorplay.float32
print(t.device)   # cpu
print(t.ndim)     # 2
```

Use `.item()` to pull a single number out of a one-element tensor, and `.numpy()` to get a
NumPy array:

```python
print(t[0, 0].item())   # 1.0
```

## Indexing, slicing and reshaping

Tensor indexing uses the same syntax as Python lists and NumPy: `[row, column]` with `:`
for slices and integers for picking positions. `...` means "every remaining dimension".

```python
t = tp.arange(12).reshape(3, 4)
print(t)                # a (3, 4) tensor
print(t[0])             # first row, shape (4,)
print(t[:, 1])          # second column, shape (3,)
print(t[1:, 1:3])       # a (2, 2) block
print(t[0, [1, 2]])     # pick columns 1 and 2 of row 0, shape (2,)
```

`reshape` changes the shape without moving the data; so does `.view`. For the common case of
removing a dimension of size 1, use `squeeze`, and for adding one use `unsqueeze`:

```python
t = tp.ones(4, 1)
print(t.squeeze().shape)  # (4,)
```

When you read an element that is itself a number you get a tensor; use `.item()` for a Python
`float` or `int`.

## Doing math: element-wise, reductions and broadcasting

Operations are element-wise by default, and a pair of tensors must broadcast to a common
shape. The rule is simple: align the shapes from the right, and each dimension can either be
equal, be 1 (it stretches), or be missing.

```python
a = tp.ones(3, 1)      # shape (3, 1)
b = tp.ones(1, 4)      # shape (1, 4)
print((a + b).shape)   # (3, 4) — the 1s stretch to match
```

Broadcastable dimensions are stretched without copying data, so `a + b` is cheap even for
large shapes. See the [broadcasting](../notes/broadcasting.md) note for the full rules.

Math you will reach for constantly:

```python
x = tp.randn(3, 5)
y = tp.randn(5, 2)
z = x @ y                               # matrix multiply, shape (3, 2)
bias = tp.zeros(1, 5)
out = x + bias                          # (3, 5) + (1, 5) -> (3, 5)
s = tp.ones(3, 5).sum()                 # total, a scalar tensor
m = tp.ones(3, 5).mean(dim=0)           # mean over one axis, shape (5,)
```

Two notes on the above:

- `@` is matrix multiplication and needs an inner dimension to match. The `matmul`
  and `mm` methods are aliases you may also see.
- `x + bias` broadcasts the `(1, 5)` bias across all 3 rows; this "add a per-feature
  bias to every sample" pattern is everywhere in neural networks.
- Reductions like `sum` and `mean` take a `dim` argument to reduce a single axis rather
  than everything, and produce the shape with that axis removed.

## Moving data around: `.to()`

A tensor lives on a device. `.to()` moves it to another device or converts its data type,
returning a new tensor.

```python
x = tp.tensor([1, 2, 3])
x = x.to(tp.float32)        # change the data type
x = x.to(tp.device('cpu'))  # move to a device (see the CUDA page for GPUs)
```

## Where to go next

- [Autograd](autograd.md) — how to make a tensor *learnable* by tracking operations for
  differentiation.
- The full list of creation, indexing, and math functions is in the
  [tensorplay API reference](../tensorplay.md).
- The [broadcasting note](../notes/broadcasting.md) explains the rules in detail.
