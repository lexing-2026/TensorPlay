# Reproducibility

If an operation depends on randomness or on non-deterministic kernels (for example some
parallel reductions, or GPU kernels), two runs may not produce bit-identical results. TensorPlay
provides a set of helpers to request deterministic algorithms and to detect when one is in use.

```python
import tensorplay as tp

print(tp.are_deterministic_algorithms_enabled())   # False by default
tp.use_deterministic_algorithms(True)
print(tp.are_deterministic_algorithms_enabled())   # True
```

`use_deterministic_algorithms` is the main switch. When enabled, TensorPlay raises an error for
operations that cannot be run deterministically, so you can find them instead of silently
getting a different answer.

```python
tp.use_deterministic_algorithms(True, warn_only=True)
```

Passing `warn_only=True` downgrades that error to a warning, which is useful for tracking down
culprits without stopping mid-run. `tp.is_deterministic_algorithms_warn_only_enabled()` reports
whether you are in that mode.

There is also a debug mode that tells you which operation is at fault:

```python
tp.set_deterministic_debug_mode(True)   # report the offending op
tp.get_deterministic_debug_mode()       # read the mode back
```

```{eval-rst}
.. autosummary::
    :nosignatures:

    tensorplay.use_deterministic_algorithms
    tensorplay.are_deterministic_algorithms_enabled
    tensorplay.is_deterministic_algorithms_warn_only_enabled
    tensorplay.set_deterministic_debug_mode
    tensorplay.get_deterministic_debug_mode
```

## Seeding for random tensors

Deterministic algorithms are about *how* an op is computed. For the random values themselves,
control the seed with `tensorplay.random` (see the [random](random.md) page).

## Where to go next

- [Randomness](notes/randomness.md) — reproducibility across runs and processes.
