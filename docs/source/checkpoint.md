```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.utils.checkpoint

```{note}
Checkpointing is implemented by rerunning each checkpointed segment's
forward pass during backward propagation. This can cause persistent states
like the RNG state to be more advanced than they would without
checkpointing. By default, checkpointing stashes and restores the RNG state
so that checkpointed passes making use of RNG (through dropout, for
example) produce the same output as non-checkpointed passes. The stash and
restore logic can incur a moderate performance hit depending on the runtime
of the checkpointed operations; if deterministic output is not required,
pass `preserve_rng_state=False` to omit it.
```

The RNG state is saved for the CPU generator and, when the checkpointed
inputs live on CUDA, for every CUDA generator referenced by them. The
device type used when no tensor carries one can be changed through
`DefaultDeviceType.set_device_type` (it defaults to `"cuda"`). The stash
happens once around the original forward; moving tensors to a different
device inside the checkpointed function cannot be anticipated, so
deterministic output is not guaranteed for such functions.

`checkpoint` may also be called without a function to produce a decorator.
This keeps checkpoint configuration separate from the arguments passed to
the checkpointed function:

```python
checkpointed_fn = checkpoint(use_reentrant=False, preserve_rng_state=False)(fn)
out = checkpointed_fn(*args, **kwargs)
```

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.utils.checkpoint.checkpoint
    tensorplay.utils.checkpoint.checkpoint_sequential
    tensorplay.utils.checkpoint.set_checkpoint_debug_enabled
    tensorplay.utils.checkpoint.CheckpointPolicy
    tensorplay.utils.checkpoint.SelectiveCheckpointContext
    tensorplay.utils.checkpoint.create_selective_checkpoint_contexts
    tensorplay.utils.checkpoint.GraphExecGroup
    tensorplay.utils.checkpoint.set_checkpoint_early_stop
    tensorplay.utils.checkpoint.set_device_states
```

