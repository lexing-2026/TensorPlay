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
loss = (model(x) - target).pow(2).mean()
loss.backward()   # loss is a scalar, fine
```

If you backpropagate from a non-scalar tensor, pass a `gradient` argument of matching shape.

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
