# tensorplay.accelerator

TensorPlay can run on CPUs and on accelerators such as GPUs. The
`tensorplay.accelerator` module exposes a small, device-agnostic surface for
asking *which accelerator is currently active*, so code that needs to know where
tensors live does not have to special-case each backend.

```python
import tensorplay as tp

dev = tp.accelerator.current_accelerator()
print(dev)                      # e.g. 'cuda:0', or None on a CPU-only machine
```

## Functions

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.accelerator.current_accelerator
```

## Where to go next

- [CUDA semantics](cuda.md) — the CUDA backend the accelerator is most often
  backed by, and its device-management functions.
- [Device concepts](tensor_attributes.md) — how device objects are spelled and
  compared.