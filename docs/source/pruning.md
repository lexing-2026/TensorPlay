```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# Pruning - tensorplay.ao.pruning

`tensorplay.ao.pruning` sparsifies a trained model: it zeroes weights that
contribute least, so the model stays the same size but computes less (with
sparse-aware kernels) or compresses better on disk. Pruning is implemented
as a *reparameterization* — the pruned weights are replaced by a mask
applied by a forward pre-hook, so the sparsity pattern stays inspectable
and reversible until you explicitly make it permanent.

Two families of methods are provided:

- **Unstructured** — zeroes individual weights chosen by importance:
  `l1_unstructured` (smallest magnitude), `random_unstructured` (uniformly
  random), `custom_from_mask` (your own mask).
- **Structured** — removes whole channels along a dimension, producing
  layout-friendly sparsity: `ln_structured` (smallest n-norm channels),
  `random_structured`.

Both take `(module, name, amount, ...)`: the module, the *name of the
parameter or buffer to prune* (e.g. `"weight"`), and how much to prune —
an int (number of entries/channels) or a float (fraction of them).

```python
import tensorplay as tp
from tensorplay.ao.pruning import l1_unstructured, ln_structured, is_pruned

model = tp.nn.Linear(64, 32, bias=False)

# zero the 30% smallest-magnitude weights
l1_unstructured(model, name="weight", amount=0.3)

# remove 10 whole output channels (dim 0 of the weight)
ln_structured(model, name="weight", amount=0.1, n=2, dim=0)

print(is_pruned(model))      # True while the mask is active
print(model.weight_mask)    # the binary mask, 1 = keep
```

## Unstructured methods

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.pruning.l1_unstructured
    tensorplay.ao.pruning.random_unstructured
    tensorplay.ao.pruning.custom_from_mask
    tensorplay.ao.pruning.global_unstructured
    tensorplay.ao.pruning.identity
    tensorplay.ao.pruning.L1Unstructured
    tensorplay.ao.pruning.RandomUnstructured
    tensorplay.ao.pruning.CustomFromMask
    tensorplay.ao.pruning.Identity
```

{func}`~tensorplay.ao.pruning.l1_unstructured` is the standard entry:
prune the `amount` smallest-|w| entries of one parameter.
{func}`~tensorplay.ao.pruning.custom_from_mask` applies a mask you supply
from domain knowledge. The module-level functions prune one parameter of
one module; {func}`~tensorplay.ao.pruning.global_unstructured` takes an
iterable of `(module, name)` pairs and a pruning method, and ranks the
entries across *all* of them together — so two layers compete for the
same global sparsity budget rather than each losing exactly `amount`%
locally. {func}`~tensorplay.ao.pruning.identity` attaches a
reparameterization that prunes nothing (an all-ones mask), useful as a
placeholder in pipelines that expect one.

## Structured methods

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.pruning.ln_structured
    tensorplay.ao.pruning.random_structured
    tensorplay.ao.pruning.LnStructured
    tensorplay.ao.pruning.RandomStructured
```

{func}`~tensorplay.ao.pruning.ln_structured` removes the channels with the
smallest L-norm along `dim` — `n` selects the norm order (2 is the common
choice; `float('inf')`, `float('-inf')`, `'fro'`, and `'nuc'` are
accepted). Structured pruning zeroes every weight of the
removed channel, so the output dimension shrinks logically — the kind of
sparsity dense kernels and downstream layers can exploit without
specialized sparse support.

## Managing the reparameterization

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.ao.pruning.remove
    tensorplay.ao.pruning.is_pruned
    tensorplay.ao.pruning.validate_pruning_amount
    tensorplay.ao.pruning.compute_nparams_to_prune
    tensorplay.ao.pruning.BasePruningMethod
    tensorplay.ao.pruning.PruningContainer
```

While pruning is active, the parameter (say `weight`) is moved aside and
the module gains `weight_orig` (the dense values), `weight_mask` (the
0/1 mask), and a forward pre-hook that recomputes `weight = weight_orig *
mask` on every call. {func}`~tensorplay.ao.pruning.remove` tears that
scaffolding down and bakes the mask into the stored values, making the
sparsity permanent and the hooks gone;
{func}`~tensorplay.ao.pruning.is_pruned` reports whether any submodule
still carries an active reparameterization.
{func}`~tensorplay.ao.pruning.validate_pruning_amount` and
{func}`~tensorplay.ao.pruning.compute_nparams_to_prune` are the helpers the
methods use to interpret `amount` and count entries.

{class}`~tensorplay.ao.pruning.BasePruningMethod` is the abstract
contract for custom methods: implement `compute_mask(t, default_mask)`
returning the new mask, and apply it with the classmethod `apply(module,
name, ...)`. {class}`~tensorplay.ao.pruning.PruningContainer` composes
several methods on one parameter — each pruning call on an
already-pruned parameter unions its mask into the container, so iterative
pruning schedules accumulate.

## Where to go next

- [quantization](quantization.md) — the other model-optimization axis:
  precision instead of sparsity. The two compose (prune, then quantize,
  or vice versa depending on the target).
- [nn](nn.md) — the modules being pruned, and `parametrizations` for the
  related constraint-style APIs.
- [the main namespace](tensorplay.md) — tensor indexing ops the mask
  math reduces to.