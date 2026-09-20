```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Computation-graph visualization - tensorplay.utils.viz

Every result produced under autograd carries the recorded chain of
operations behind it. `make_dot` walks that chain and renders it as a
picture, so you can see exactly which operations produced a value before
calling backward on it.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.utils.viz.make_dot
```

## Rendering a graph

```python
import tensorplay as tp
from tensorplay.utils.viz import make_dot

x = tp.randn(5, 5, requires_grad=True)
w = tp.randn(5, 5, requires_grad=True)
loss = ((x @ w).relu()).sum()

dot = make_dot(loss, params={"x": x, "w": w})
dot.render("graph", format="png")   # writes graph.png in the working directory
```

Node colors carry the structure:

* light blue — leaf tensors with ``requires_grad=True``; those listed in
  ``params`` are labeled by the name you passed,
* light green — the output tensor handed to `make_dot`,
* white — the recorded operations, labeled by operation name.

## Render backends

`make_dot` picks whichever backend is installed:

* the ``graphviz`` package returns a real ``graphviz.Digraph`` — call
  ``.render(filename, format=...)`` to write an image, or ``.pipe()`` for
  the raw bytes. Rasterizing to a file also needs the ``dot`` program on
  your ``PATH``.
* otherwise ``networkx`` + ``matplotlib`` draw a hierarchical layout and
  return a wrapper with the same ``.render(filename, format="png")`` call.
* with neither installed, calling `make_dot` raises ``RuntimeError``.

## Reading the graph as text

The same chain is walkable without any rendering dependency: every tensor
with a gradient history exposes its ``grad_fn``, and every recorded
operation exposes its name and the operations it consumed.

```python
fn = loss.grad_fn
while fn is not None:
    print(fn.name)
    fn = fn.next_functions[0][0] if fn.next_functions else None
```
