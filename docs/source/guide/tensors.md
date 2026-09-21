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

## Data types

Every tensor carries a dtype. The floating-point types are `float32` (the default for
`tp.tensor`, `tp.randn`, and the other float constructors), `float64` (double precision, for
when 32 bits are not enough — careful numerical work, physics, finance), and the reduced
precision `float16` and `bfloat16` common on accelerators. Integer tensors come in
`int8`/`int16`/`int32`/`int64` plus `uint8`, and there is `bool` and the complex pair
`complex64`/`complex128`:

```python
i = tp.tensor([1, 2, 3])          # int64 — integer input defaults to the widest int type
f = tp.tensor([1.0, 2.0])         # float32 — the floating-point default
bf = tp.tensor([1.5], dtype=tp.bfloat16)

print(i.dtype)        # tensorplay.int64
print(f.dtype)        # tensorplay.float32
print((bf * 2).dtype) # tensorplay.bfloat16 — arithmetic stays in the input type
```

Mixed-type arithmetic promotes to the "wider" type, and `.to(dtype)` converts explicitly:

```python
a = tp.tensor([1, 2, 3], dtype=tp.int64)
print((a + tp.ones(3, dtype=tp.float32)).dtype)  # tensorplay.float32 — float wins over int
print((a + tp.ones(3, dtype=tp.int32)).dtype)    # tensorplay.int64 — the wider int wins
print(a.to(tp.float16).dtype)                    # tensorplay.float16
```

For most deep-learning work `float32` is the right default: it halves memory and bandwidth
compared to `float64`, and it is what the compute kernels are tuned for.

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

### Masks and index tensors

A boolean tensor of the same shape selects the elements where it is `True`, and
`masked_fill` replaces the selected elements while keeping the shape:

```python
t = tp.tensor([1.0, -2.0, 3.0, -4.0])
print(t[t > 0])                       # [1.0, 3.0]
print(t.masked_fill(t < 0, 0.0))      # [1.0, 0.0, 3.0, 0.0]
```

`tp.where(cond, a, b)` is the element-wise "pick from `a` or `b`" counterpart that always
keeps the shape, and `nonzero` returns the integer positions of the truthy entries:

```python
print(tp.where(t > 0, t, tp.zeros_like(t)))  # [1.0, 0.0, 3.0, 0.0]
print(t.nonzero().flatten().tolist())        # [0, 2]
```

Two index-based operations show up constantly once you implement anything with lookups —
embeddings, attention, top-k selection: `index_select` picks whole rows or columns, and
`gather` collects one element per index along an axis:

```python
m = tp.arange(12).reshape(3, 4)
print(m.index_select(0, tp.tensor([2, 0])))     # rows 2 and 0, in that order
print(m.gather(1, tp.tensor([[0], [2], [1]])))  # one picked column per row
```

## Views and copies

Several operations return a *view* — a new tensor header that shares the same underlying
storage: slicing, `view`, `reshape` (when the input is contiguous), `squeeze`, `unsqueeze`,
`permute`, and `expand`. Writing through a view writes the original:

```python
t = tp.arange(6)
v = t.view(2, 3)
v[0, 0] = 99
print(t.tolist())    # [99, 1, 2, 3, 4, 5]
```

`expand` stretches a size-1 dimension without copying — it is the machinery behind
broadcasting — while `repeat` is the copying counterpart:

```python
col = tp.arange(3).reshape(3, 1)
print(col.expand(3, 4).shape)   # (3, 4) — no copy
print(col.repeat(1, 4).shape)   # (3, 4) — a real copy
```

`reshape` returns a view only when the data is already laid out contiguously for the new
shape; after a `permute` it silently copies instead:

```python
t = tp.arange(6).reshape(2, 3)
p = t.permute(1, 0)    # shape (3, 2), non-contiguous layout
r = p.reshape(6)       # still works, but this time it copies
```

When you want a guaranteed independent copy, call `.clone()`. When a kernel requires the
memory to be dense in the new order, call `.contiguous()`.

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

Two reduction-adjacent helpers worth knowing: `max`/`min` (which also return *where* the
extreme was, via `argmax`/`argmin`), and `topk` for the k largest:

```python
x = tp.tensor([3.0, 1.0, 4.0, 1.5])
print(x.argmax().item())              # 2 — position of the maximum
values, indices = x.topk(2)           # returns a (values, indices) pair
print(values)                         # [4.0, 3.0] — the two largest, sorted
```

Special values are part of float arithmetic: `nan` (not a number) and `inf` can appear from
`0/0` or overflow. `isnan` and `isinf` locate them, `clamp` bounds a range, and `nan_to_num`
replaces them with finite stand-ins:

```python
x = tp.tensor([float('nan'), 1.0, float('inf')])
print(x.isnan().tolist())                          # [True, False, False]
print(tp.nan_to_num(x, nan=0.0, posinf=99.0))      # [0.0, 1.0, 99.0]
print(tp.tensor([1.0, -5.0, 9.0]).clamp(-2, 2))    # [1.0, -2.0, 2.0]
```

## Combining tensors: `cat` and `stack`

`cat` joins tensors along an existing dimension; `stack` adds a new one:

```python
a = tp.ones(2, 3)
b = tp.zeros(2, 3)
print(tp.cat([a, b]).shape)         # (4, 3) — glued along dim 0
print(tp.cat([a, b], dim=1).shape)  # (2, 6)
print(tp.stack([a, b]).shape)       # (2, 2, 3) — a new leading dimension
```

`cat` requires matching shapes on every other dimension; `stack` requires identical shapes.
Use `stack` when the dimension you are building *is* the batch: three per-sample results of
shape `(3, 5)` become a `(3, 3, 5)` batch with `tp.stack([r0, r1, r2])`.

## In-place operations

Operations whose method name ends in an underscore modify the tensor instead of returning a
new one:

```python
x = tp.ones(3)
x.add_(1)         # x is now [2, 2, 2]
x.mul_(10)        # [20, 20, 20]
x.clamp_(max=15)  # [15, 15, 15]
print(x.tolist())
```

In-place operations save allocation in the middle of a long pipeline. Under autograd there
is one rule to respect: a tensor that requires a gradient and is marked as a *leaf* (one you
created, not one produced by an operation) may not be modified in place — TensorPlay raises
an error telling you exactly that, so you cannot corrupt a gradient silently. The standard
`x = x + 1` form stays out-of-place and is always safe.

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
