# tensorplay.func whirlwind tour

```{eval-rst}
.. currentmodule:: tensorplay.func
```

A hands-on pass over the transforms, in the order you would usually meet
them. Every snippet is self-contained and can be run as-is.

## {func}`grad` — gradient computation

`grad(func)` returns a new function that computes the gradient of `func`.
It assumes `func` returns a single-element tensor and by default
differentiates with respect to the first argument.

```python
import tensorplay as tp
from tensorplay.func import grad

x = tp.randn([])
cos_x = grad(lambda x: tp.sin(x))(x)
assert tp.allclose(cos_x, x.cos())

# second-order gradients are just grad of grad
neg_sin_x = grad(grad(lambda x: tp.sin(x)))(x)
assert tp.allclose(neg_sin_x, -x.sin())
```

## {func}`vmap` — auto-vectorization

`vmap(func)` returns a new function that maps `func` over a dimension
(default: 0) of each tensor input. Write the function for a single sample;
`vmap` handles the batch:

```python
import tensorplay as tp
from tensorplay.func import vmap

batch_size, feature_size = 3, 5
weights = tp.randn(feature_size, requires_grad=True)

def model(feature_vec):
    assert feature_vec.dim() == 1
    return (feature_vec * weights).sum().relu()

examples = tp.randn(batch_size, feature_size)
result = vmap(model)(examples)
assert result.shape == (batch_size,)
```

For a pure function, `vmap(f)(x)` is equivalent to stacking one call per
sample:

```python
import tensorplay as tp
from tensorplay.func import vmap

xs = tp.randn(4, 3)
assert tp.allclose(
    vmap(lambda x: x * x)(xs),
    tp.stack([x * x for x in xs.unbind(0)]),
)
```

{func}`vmap` imposes restrictions on the code it can map over — see
{ref}`ux-limitations`.

### Per-sample gradients

Composing `vmap` and `grad` computes per-sample gradients: each sample's
gradient is produced independently, with no averaging across the batch.

```python
import tensorplay as tp
from tensorplay.func import grad, vmap

batch_size, feature_size = 3, 5

def compute_loss(weights, example, target):
    y = (example * weights).sum().relu()
    return (y - target) ** 2

weights = tp.randn(feature_size, requires_grad=True)
examples = tp.randn(batch_size, feature_size)
targets = tp.randn(batch_size)

grad_per_example = vmap(grad(compute_loss), in_dims=(None, 0, 0))(
    weights, examples, targets
)
assert grad_per_example.shape == (batch_size, feature_size)
```

The `in_dims=(None, 0, 0)` says the weights are shared across the batch
while the examples and targets are mapped.

## {func}`vjp` — vector-Jacobian product

`vjp` applies `func` to its inputs and returns a new function that computes
the vector-Jacobian product for given cotangents:

```python
import tensorplay as tp
from tensorplay.func import vjp

inputs = tp.randn(3)
outputs, vjp_fn = vjp(tp.sin, inputs)
vjps = vjp_fn(tp.randn(3))
```

## {func}`jvp` — Jacobian-vector product

`jvp` computes forward-mode AD. Unlike most other transforms it is not a
higher-order function: it returns the outputs of `func(inputs)` together
with the Jacobian-vector products for the given tangents:

```python
import tensorplay as tp
from tensorplay.func import jvp

x, y = tp.randn(5), tp.randn(5)
_, out_tangent = jvp(lambda x, y: x * y, (x, y), (tp.ones(5), tp.ones(5)))
assert tp.allclose(out_tangent, x + y)
```

## {func}`jacrev`, {func}`jacfwd`, and {func}`hessian`

`jacrev` returns a new function that takes in `x` and returns the Jacobian
using reverse-mode AD; `jacfwd` is its forward-mode counterpart:

```python
import tensorplay as tp
from tensorplay.func import jacfwd, jacrev

x = tp.randn(5)
assert tp.allclose(jacrev(tp.sin)(x), tp.diag(tp.cos(x)))
assert tp.allclose(jacfwd(tp.sin)(x), tp.diag(tp.cos(x)))
```

Composing the two directions produces Hessians, and {func}`hessian` is the
convenience wrapper:

```python
import tensorplay as tp
from tensorplay.func import hessian, jacfwd, jacrev

def f(x):
    return x.sin().sum()

x = tp.randn(5)
h0 = jacrev(jacrev(f))(x)
h1 = jacfwd(jacrev(f))(x)
h2 = hessian(f)(x)
assert tp.allclose(h0, h2) and tp.allclose(h1, h2)
```

## {func}`linearize`

`linearize` evaluates `func` once and returns the value together with a
callable forward-mode linearization at that point:

```python
import tensorplay as tp
from tensorplay.func import linearize

x = tp.randn(5)
y, jvp_fn = linearize(tp.sin, x)
assert tp.allclose(jvp_fn(tp.ones(5)), tp.cos(x))
```

## What composes today

Not every composition of transforms is available yet. The verified
compositions:

- `vmap(grad(f))` with a shared differentiable argument — the per-sample
  gradient pattern above.
- `vmap(vmap(f))` — mapping over two independent batch dimensions.
- `grad(vmap(f))` — differentiating a batched evaluation.

`vmap(jacrev(f))` and `vmap(jacfwd(f))` — batched Jacobians — currently
raise; see {ref}`ux-limitations` for the details.
