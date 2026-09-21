# foreach operations

`foreach` operations lift an ordinary tensor operation over a whole list of tensors in
one call. `tensorplay._foreach_add(tensors, other)` is semantically equivalent to a
Python loop applying `tensorplay.add` at each list position — but it runs as one
multi-tensor kernel when the backend and the operands allow it, and it is the primitive
the built-in optimizers are built on: applying an update to every parameter of a model
is exactly a foreach operation.

```python
import tensorplay as tp

xs = [tp.ones(3) * 1, tp.ones(3) * 2, tp.ones(3) * 3]
ys = [tp.ones(3)] * 3

# out-of-place: returns a new list, inputs untouched
print([t.tolist() for t in tp._foreach_add(xs, ys)])
# [[2.0, 2.0, 2.0], [3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]

# in-place: the trailing underscore, like the scalar-op convention
tp._foreach_mul_(xs, 2.0)
print([t.tolist() for t in xs])
# [[2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]]

# reductions lift too
print([n.item() for n in tp._foreach_norm(xs)])
# [3.464..., 6.928..., 10.392...]
```

## Naming and coverage

The functions live directly on the `tensorplay` namespace with a `_foreach_` prefix:
`tensorplay._foreach_add`, `tensorplay._foreach_mul_`, `tensorplay._foreach_norm`, and
about ninety more, covering the unary math functions (`_foreach_abs`, `_foreach_acos`,
`_foreach_ceil`, ...), the binary arithmetic (`_foreach_add`, `_foreach_sub`,
`_foreach_div`, `_foreach_clamp_min`, ...) in both out-of-place and in-place spellings,
plus list-level utilities (`_foreach_clone`, `_foreach_copy_`, `_foreach_norm`,
`_foreach_zero_`), matrix multiply over lists (`_foreach_mm`), and the fused optimizer
kernels (`_foreach_adam`, `_foreach_sgd`).

## Operand forms

The second operand of a binary foreach op can be:

- a **TensorList** of the same length — position-wise, each pair together;
- a **single Tensor** — shared across every position;
- a **Scalar or list of Scalars** — the elementwise-constant form.

```python
tp._foreach_add(xs, ys)            # TensorList / TensorList
tp._foreach_add(xs, ys[0])         # TensorList / Tensor (shared)
tp._foreach_add(xs, 2.0)           # TensorList / Scalar
tp._foreach_add(xs, [1.0, 2.0, 3.0])   # TensorList / list of Scalars
```

Tensor lists must be non-empty, and corresponding tensor/scalar lists must match in
length. On CUDA the fused multi-tensor path additionally requires tensors on the same
device with compatible dtypes and layouts; where a list does not qualify, results stay
correct — only the single-kernel speedup is lost.

## When to reach for them

Hand-written training code rarely needs them — the optimizers already use the fused
forms internally. Direct use pays off when you maintain parallel lists of tensors
outside an optimizer: applying the same elementwise fixup to a list of buffers,
zeroing a list of gradients, or computing per-tensor statistics in one call.
