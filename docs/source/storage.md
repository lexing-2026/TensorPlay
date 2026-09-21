# Storage

A tensor is a view over a one-dimensional block of memory. Concretely, a tensor is
defined by:

- **Storage**: the actual data, as a contiguous one-dimensional array of bytes.
- `dtype`: the data type of the elements.
- **shape**: the size in each dimension.
- **stride**: how many elements to step in the storage when moving one position along
  each dimension.
- **offset**: where in the storage the tensor's data starts (zero for freshly created
  tensors).

The tensor is the metadata; the storage is the payload. This split is what lets many
tensors share memory: a view (from `view`, `reshape` on a contiguous tensor, slicing, or
`expand`) is a *new tensor header pointing at the same storage*.

## Untyped Storage

{meth}`tensorplay.Tensor.untyped_storage` returns the storage as an
{class}`tensorplay.UntypedStorage` — a flat byte array that is deliberately *untyped*:
it knows its size in bytes and its device, not the element type of any particular tensor
viewing it.

```python
import tensorplay as tp

t = tp.arange(6)              # int64: 6 elements * 8 bytes
s = t.untyped_storage()
print(s.size())               # 48 — bytes
print(s.nbytes())             # 48
print(s.device)               # cpu
```

Storage identity is how you check whether two tensors share memory: same storage
`data_ptr()` means same memory. Views keep it; `clone()` buys new storage:

```python
t = tp.arange(6)
print(t.view(2, 3).untyped_storage().data_ptr() == t.untyped_storage().data_ptr())
# True — a view is a new header over the same bytes
print(t[1:].untyped_storage().data_ptr() == t.untyped_storage().data_ptr())
# True — slicing is a (offset, stride) change, not a copy
print(t.clone().untyped_storage().data_ptr() == t.untyped_storage().data_ptr())
# False — clone allocates
```

Note that a tensor's own `data_ptr()` points at *its first element* (storage start plus
its storage offset times element size), so a sliced tensor's `data_ptr` differs from its
storage's even though the storage is shared — compare storages, not tensor pointers,
when the question is "same memory?".

## API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.UntypedStorage
```

{class}`tensorplay.UntypedStorage` exposes:

- `size()` / `nbytes()` — the byte length (one method each; they agree).
- `data_ptr()` — the address of the first byte.
- `device` — where the storage lives.
- `is_cuda` — whether it is CUDA memory.
- `resizable()` — whether the storage may grow, and `resize_(new_size_bytes)` to
  change the byte length in place:

```python
s = tp.arange(4).untyped_storage()   # int64: 32 bytes
print(s.resizable())                 # True
s.resize_(16)                        # shrink to 16 bytes
print(s.size())                      # 16
```

Resizing storage that tensors still view is a low-level operation — the viewing tensors
keep their old shapes and offsets, which may now run past the end. It exists for the
serialization layer and allocator internals; ordinary code should build a new tensor
instead.
