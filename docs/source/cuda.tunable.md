# tensorplay.cuda.tunable

```{eval-rst}
.. automodule:: tensorplay.cuda.tunable
```

TunableOp is the mechanism that records, for each shape of a small set of
runtime-tunable operations (GEMM above all), which underlying kernel implementation is
fastest on the current hardware — and replays that choice on later runs instead of
re-searching. The tuning results are written to a results file that can be read back, so
a machine is tuned once and reused.

This TensorPlay build does not include the runtime instrumentation TunableOp requires.
The control surface below is complete — every function is importable and has its
documented signature — but tuning is permanently off: `is_enabled()` always returns
`False`, and any call that would turn tuning on raises
`RuntimeError: TunableOp is not supported by this TensorPlay build`.

```python
from tensorplay.cuda import tunable

print(tunable.is_enabled())   # False — the honest answer in this build
try:
    tunable.enable()
except RuntimeError as e:
    print(e)                  # TunableOp is not supported by this TensorPlay build
```

## API

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.cuda.tunable.enable
    tensorplay.cuda.tunable.disable
    tensorplay.cuda.tunable.is_enabled
    tensorplay.cuda.tunable.tuning_enable
    tensorplay.cuda.tunable.tuning_disable
    tensorplay.cuda.tunable.tuning_is_enabled
    tensorplay.cuda.tunable.record_untuned_enable
    tensorplay.cuda.tunable.record_untuned_disable
    tensorplay.cuda.tunable.record_untuned_is_enabled
    tensorplay.cuda.tunable.set_verbose
    tensorplay.cuda.tunable.is_verbose
    tensorplay.cuda.tunable.set_max_tuning_duration
    tensorplay.cuda.tunable.get_max_tuning_duration
    tensorplay.cuda.tunable.set_max_tuning_samples
    tensorplay.cuda.tunable.get_max_tuning_samples
    tensorplay.cuda.tunable.read_file
    tensorplay.cuda.tunable.write_file
```

The `tuning_*` pair controls whether results are written to the results file as they are
measured, the `record_untuned_*` pair controls logging of operations that ran without a
tuned choice, the `*_max_tuning_*` pair bounds how long and how many samples a single
tuning search may take, and `read_file` / `write_file` load or persist the results
database.
