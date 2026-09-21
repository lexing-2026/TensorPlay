# Autograd

Automatic differentiation is what makes training a neural network possible without you
deriving gradients by hand. In TensorPlay you create tensors with `requires_grad=True`, run
your forward computation, call `.backward()`, and read the gradient off each tensor's
`.grad`.

```python
import tensorplay as tp

x = tp.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x * x).sum()
y.backward()
print(x.grad)  # tensor([2., 4., 6.])
```

Here `y = x² sum`, so the gradient of `y` with respect to `x` is `2x`. The printed gradient
shows `backward()` worked.

## How it works

When you do math on a tensor that `requires_grad`, TensorPlay records each operation into a
graph: the output remembers which inputs and which operation produced it. A tensor that was
produced by an operation has a `grad_fn`, the node that can recompute partial derivatives.
`backward()` starts at the scalar you call it on and walks that graph outward, applying the
chain rule node by node.

```python
x = tp.tensor([2.0], requires_grad=True)
w = tp.tensor([3.0], requires_grad=True)
z = x * w + x
print(z.grad_fn)  # not None — z knows how it was made
z.backward()
print(x.grad)  # (w + 1) = 4.0
print(w.grad)  # x = 2.0
```

Only tensors with `requires_grad=True` accumulate a gradient. Any tensor produced from them
will *also* ask for a gradient by default, which is what lets `backward()` flow through a
whole model.

## Reading gradients

```
x.grad
```

`.grad` holds the gradient of the scalar you called `backward()` on, with respect to `x`.
The value is accumulated, so if you call `backward()` more than once without zeroing, the
gradients add up. Modern training code resets them at the start of each step — see
[Training](training.md).

## Turning off tracking

Tracking comes at a cost and is usually not what you want when running a forward pass only,
for example when evaluating a model. `no_grad` disables tracking for everything inside it:

```python
x = tp.tensor([1.0, 2.0, 3.0], requires_grad=True)

with tp.no_grad():
    y = x * 2
print(y.requires_grad)  # False — no graph was built, faster
```

`tensorplay.set_grad_enabled(False)` and `tensorplay.enable_grad()` do the same thing in a
non-context-manager form.

A related method is `.detach()`, which returns a new tensor that shares the same data but is
cut off from the graph — its `requires_grad` is `False` and `backward()` will not flow
through it. Use it when you want to use a computed value as a constant:

```python
x = tp.tensor([2.0], requires_grad=True)
const = (x * 3).detach()
print(const.requires_grad)  # False — shared data, but detached from the graph
```

## What `backward()` needs

`backward()` must be called on a scalar (a tensor with a single element), or you must pass
the initial gradient you want it to use:

```python
x = tp.tensor([1.0, 2.0], requires_grad=True)
target = tp.tensor([2.0, 2.0])
loss = ((x * 2 - target) ** 2).mean()
loss.backward()   # loss is a scalar, fine
```

If you backpropagate from a non-scalar tensor, pass a `gradient` argument of matching shape.

```python
x = tp.tensor([1.0, 2.0, 3.0], requires_grad=True)
z = x * x                      # shape (3,) — not a scalar
z.backward(tp.ones_like(z))     # tell backward the weights of each entry
print(x.grad)                  # tensor([2., 4., 6.])
```

The `gradient` argument is the vector in the vector-Jacobian product: `.backward()` computes
`grad_output @ J` where `J` is the Jacobian of the output with respect to the inputs. For a
scalar loss the implicit vector is `1`, which is why plain `loss.backward()` needs nothing.

## `tp.autograd.grad`: gradients without storing them

`.backward()` writes into `.grad` as a side effect. When you want gradients as *values* —
inside a computation, without touching any `.grad` field — use `tp.autograd.grad`:

```python
x = tp.tensor([1.0, 2.0], requires_grad=True)
y = (x * x).sum()
(g,) = tp.autograd.grad(y, x)
print(g)          # tensor([2., 4.])
print(x.grad)     # None — nothing was written
```

It takes the outputs, the inputs to differentiate with respect to, and returns one gradient
per input.

## Accumulation, and reusing the graph

Gradients accumulate: each `.backward()` *adds* into `.grad`, which is what makes gradient
accumulation over several small batches work — but it also means a stale `.grad` silently
pollutes the next step, hence the `opt.zero_grad()` in every training loop.

```python
x = tp.tensor([3.0], requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)
print(x.grad)     # tensor([27.])
y.backward(retain_graph=True)
print(x.grad)     # tensor([54.]) — the two runs added up
```

By default the graph is freed once `backward()` has walked it. A second `backward()` on the
same output is accepted but does nothing — there is no graph left to walk, so `.grad` keeps
whatever the first run left there. Pass `retain_graph=True` when you need to differentiate
the same computation more than once — gradient penalties, comparing loss weightings — and
reset `x.grad` between runs if the accumulation is not wanted.

## Second-order gradients

`backward()` and `autograd.grad` accept `create_graph=True`, which builds the graph *for the
gradient computation itself*. Differentiating that again gives you second derivatives:

```python
x = tp.tensor([3.0], requires_grad=True)
y = x ** 3
(dy_dx,) = tp.autograd.grad(y, x, create_graph=True)   # 3x², itself differentiable
dy_dx.backward()
print(x.grad)     # tensor([18.]) — d²y/dx² = 6x
```

This is the machinery behind gradient penalty regularization and some meta-learning methods;
it costs roughly another level of graph, so use it where the math actually needs it.

## Hooks: inspecting or rewriting gradients in flight

`tensor.register_hook(fn)` calls `fn(grad)` with the gradient as it flows past that tensor
during `backward()`. The hook's return value replaces the gradient, so you can debug (print,
assert finiteness) or modify (scale, clip, zero out) on the fly:

```python
x = tp.tensor([2.0], requires_grad=True)
y = x * 2
y.register_hook(lambda grad: grad * 10)
y.backward()
print(x.grad)     # tensor([20.]) — the hook scaled the gradient tenfold
```

Hooks fire only during backward, in reverse order of the forward computation, and only for
tensors that require grad.

## `inference_mode`

`no_grad` has a stricter sibling. Inside `tp.inference_mode()`, operations also skip
version-counter bookkeeping, which makes the forward pass marginally cheaper still — the
trade-off is that tensors produced there can never be used in autograd later, even after the
block exits:

```python
x = tp.tensor([1.0], requires_grad=True)
with tp.inference_mode():
    y = x * 2
print(y.requires_grad)    # False
```

Use `no_grad` when the values might feed back into training (e.g. target computation); use
`inference_mode` for pure inference or data preprocessing at the edge of your program.

## A manual chain-rule check

To see the graph rather than trust it, you can compute a gradient by hand and compare:

```python
x = tp.tensor([3.0], requires_grad=True)
y = x ** 3
y.backward()
print(x.grad)         # 3 * x**2 = 27.0
```

## Where to go next

- [Models](models.md) — connect these learnable tensors into a network that manages its own
  parameters.
- The [autograd note](../notes/autograd.md) is a much deeper walkthrough of the engine,
  including how to define custom operations that participate in the graph.
- The [gradcheck note](../notes/gradcheck.md) shows how to programmatically verify that a
  custom gradient is correct.
