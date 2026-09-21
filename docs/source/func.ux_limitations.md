```{eval-rst}
.. currentmodule:: tensorplay.func
```

(ux-limitations)=

# UX limitations

The transforms work best on pure functions: functions whose output is
completely determined by their inputs and that do not perform side effects.
Certain in-place operations are also supported. Writing code compatible
with the transforms may involve changing habits, but in exchange the
transforms let you express quantities — per-sample gradients, ensembles,
higher-order derivatives — that are awkward to compute any other way.

## General limitations

All transforms share one rule: everything you want out of a function must
be *returned* from it. Assigning to a global variable (or to a mutable
object captured from outside) does compute — the assignments happen — but
the transform does not track them, so they cannot be differentiated or
mapped over.

So, instead of the following:

```python
import tensorplay as tp
from tensorplay.func import grad, vmap

intermediate = None

def f(x):
    global intermediate
    intermediate = x.sin()      # computed, but invisible to the transform
    return intermediate.sin()

x = tp.randn([])
grad_x = grad(f)(x)             # fine, but `intermediate` is not a result
```

return the intermediate and mark it as auxiliary data:

```python
def f(x):
    intermediate = x.sin()
    return intermediate.sin(), intermediate

grad_x, intermediate = grad(f, has_aux=True)(x)
```

## `tensorplay.autograd` APIs

Using `tensorplay.autograd.grad` or `Tensor.backward` *inside* a function
being transformed is not supported. Under {func}`vmap` the call raises
`IndexError` (the batched tensor does not survive the plain-autograd
boundary), and under {func}`grad` the graph is silently severed — the inner
call runs on detached tensors and the outer gradient comes out zero. Use
the transform equivalents instead:

- `tensorplay.autograd.grad`, `Tensor.backward` → {func}`vjp` or {func}`grad`
- forward-mode products → {func}`jvp`
- Jacobians → {func}`jacrev` or {func}`jacfwd`
- Hessians → {func}`hessian`

## vmap limitations

{func}`vmap` is the most restrictive transform. The gradient-related
transforms ({func}`grad`, {func}`vjp`, {func}`jvp`) do not have the
restrictions listed here; {func}`jacfwd` and {func}`hessian` build on
forward-mode machinery and are subject to their own coverage gaps.

`vmap(func)` returns a function that maps `func` over some new dimension of
each tensor input. The mental model is a for-loop: for pure functions,
`vmap(f)(x)` is equivalent to `tp.stack([f(x_i) for x_i in x.unbind(0)])`.

### Operator coverage

{func}`vmap` executes `func` once over batched inputs, so every operator
`func` calls needs a batching rule. An operator without one raises
`NotImplementedError` naming the operator, for example:

```python
vmap(lambda a, b: a.dot(b))(tp.randn(3, 4), tp.randn(3, 4))
# NotImplementedError: Kernel not found for op: dot on backend: VmapCPU
```

Verified to work under `vmap`: the arithmetic operators (`+`, `-`, `*`, `/`,
`**`, unary `-`), the elementwise math functions (`abs`, `exp`, `log`,
`sqrt`, `sin`, `cos`, `tanh`, `sigmoid`, `relu`), `matmul`, `mm`, `bmm`,
`sum`, `cumsum`, `clamp`, `where`, `stack`, `cat`, `index_select`,
`narrow`, `select`, `tril`, `maximum`, `minimum`, `logsumexp`, the shape
operations (`reshape`, `view`, `transpose`, `permute`, `movedim`,
`squeeze`, `unsqueeze`, `expand`, `contiguous`), basic and advanced
indexing, the `new_*` factories plus `tp.zeros`/`tp.ones`, and `randn`.

Not yet covered — they raise the `NotImplementedError` above: `dot`,
`mean`, `max`, `min`, `argmax`, `softmax`, `log_softmax`, `norm`,
`masked_fill`, `clone`, `sort`, `argsort`, `topk`, `einsum`, `dropout`,
`randn_like`, the norm layers (`batch_norm`, `group_norm`, `layer_norm`),
`nonzero`, and `item`.

:::{warning}
`vmap` over a `Linear` module — or over `functional.linear` — with a
*batched input* currently mis-shapes the output: the result arrives with an
extra leading dimension (each slice along it identical), instead of the
stacked per-sample outputs. Until the rule propagates the batch tag,
formulate the layer explicitly — `x @ W + b` maps correctly — or map over
the *parameters* instead of the inputs, which is the
{func}`stack_module_state` ensemble pattern and works as documented.
:::

### In-place operations

In-place arithmetic has no batching rule and raises cleanly:

```python
vmap(lambda x, y: x.add_(y), in_dims=(0, 0))(tp.randn(3, 1), tp.randn(3, 1))
# NotImplementedError: Kernel not found for op: add_.Tensor on backend: VmapCPU
```

This holds whether the mutated tensor is batched or not — prefer the
out-of-place form under `vmap`, or `x + y` directly.

### Data-dependent operations

`.item()` and `nonzero` raise `NotImplementedError` under `vmap`; the
output of a data-dependent op varies per sample, so no single stacked
result exists. Rewrite the code to avoid materializing per-sample
shapes.

### Data-dependent Python control flow

:::{warning}
The condition of an `if` (or a `while`/`for` test) must not be a tensor
being mapped over. In the current build this does not fail gracefully —
it **crashes the interpreter** (a fault in the layer that converts a
batched tensor to a boolean), so treat it as a hard error:

```python
def relu(x):
    if x > 0:      # x is a mapped tensor here: do not do this
        return x
    return 0 * x

vmap(relu)(tp.randn(3))     # crashes the process
```

Re-express value-dependent branches with {func}`tp.where`, whose
comparison and selection rules are supported.
:::

### The `out=` keyword

The public operator signatures do not accept `out=` (passing one raises
`TypeError` about the argument combination), so there is no `out=` form
for `vmap` to transform.

### Randomness

The intent of a random operation under `vmap` is ambiguous — should every
sample see fresh values, or the same ones? — so `vmap` takes a
`randomness` flag instead of guessing:

- `"error"` (the default): any random operation raises
  `RuntimeError: random operations are not allowed under randomness=error`.
- `"different"`: elements of the batch draw different values.
- `"same"`: every element of the batch sees the same value.

```python
def add_noise(x):
    return x + tp.randn(())

x = tp.ones(3)
r = vmap(add_noise, randomness="same")(x)
assert (r == r[0]).all()          # one value, repeated

r = vmap(add_noise, randomness="different")(x)
assert (r != r[0]).any()          # fresh value per sample
```

An invalid value raises before the call: only `"error"`, `"different"`,
and `"same"` are accepted. The flag governs the tensor factories such as
`tp.randn`; `dropout` and `randn_like` do not have batching rules yet and
raise regardless of the flag.

## Composability

Verified compositions of the transforms with each other:

| Composition | Status |
| --- | --- |
| `vmap(grad(f))`, differentiable argument shared via `in_dims=None` | works — per-sample gradients |
| `vmap(grad(f))`, mapping over the *differentiated* argument | raises `IndexError` |
| `vmap(vmap(f))` | works — two independent batch dimensions |
| `grad(vmap(f))` | works |
| `vmap(jacrev(f))` | raises — the internal cotangent pass loses the batch shape |
| `vmap(jacfwd(f))` | raises — the forward-mode rule for `zeros_like` is missing |

## Norm layers

`batch_norm`, `group_norm`, and `layer_norm` all lack batching rules, so
{func}`vmap` over any of them raises. Worse, the reverse-mode Jacobian
transforms do not fail loudly on batch norm — they return garbage values
(order 1e33, or NaN once running stats are disabled). See
[patching batch norm](func.batch_norm.md) before transforming a model that
contains normalization.
