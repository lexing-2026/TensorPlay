# tensorplay.backends

```{eval-rst}
.. automodule:: tensorplay.backends
```

`tensorplay.backends` exposes per-library controls: which math libraries this build
links against, whether they are available, and the library-specific knobs that change
how kernels run. The submodules are `cpu`, `cuda`, `cudnn`, `mkl`, `mkldnn`, `nnpack`,
and `openmp`.

## tensorplay.backends.cpu

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.backends.cpu.get_cpu_capability
```

{func}`tensorplay.backends.cpu.get_cpu_capability` reports the highest SIMD capability
the CPU dispatch layer selects for (`"AVX2"`, `"AVX512"`, ...). Kernels compiled for
several instruction sets dispatch on this value, so it is the first thing to check when
CPU throughput looks wrong.

## tensorplay.backends.cuda

Controls for the CUDA libraries the CUDA device layer links. `cuBLASModule` is the cuBLAS
handle module; `cuFFTPlanCache` is the per-device plan cache for FFTs, with `clear()`,
`size()`, and `max_size` for managing it — plans are expensive to build, and the cache
is what makes repeated FFTs cheap.

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.backends.cuda.allow_fp16_bf16_reduction_math_sdp
    tensorplay.backends.cuda.can_use_flash_attention
    tensorplay.backends.cuda.can_use_efficient_attention
    tensorplay.backends.cuda.can_use_cudnn_attention
    tensorplay.backends.cuda.cuBLASModule
    tensorplay.backends.cuda.cuFFTPlanCache
```

- {func}`tensorplay.backends.cuda.allow_fp16_bf16_reduction_math_sdp` toggles whether
  the math implementation of scaled-dot-product attention may accumulate its reduction
  in fp16/bf16 (enabled or disabled as a plain call; pass `False` when numerical checks
  require full fp32 reductions).
- The `can_use_*` predicates take an `SDPAParams` record and report whether the
  corresponding attention kernel would accept it — the same probes the dispatcher
  consults (see [attention](nn.attention.md)).

## tensorplay.backends.cudnn and tensorplay.backends.mkldnn

Both are thin re-export modules (`m`) for the cuDNN and oneDNN-style dense-CPU library
bindings compiled into this build. They exist so backend-specific code can be written
against a stable import path.

## tensorplay.backends.mkl

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.backends.mkl.is_available
```

{func}`tensorplay.backends.mkl.is_available` reports whether the CPU math library is
linked and usable.

## tensorplay.backends.nnpack

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.backends.nnpack.is_available
    tensorplay.backends.nnpack.flags
    tensorplay.backends.nnpack.set_flags
```

`nnpack` is the packing-based CPU convolution path. `flags(enabled=...)` is the
context-manager form — the setting applies inside the `with` block and reverts on exit —
and `set_flags` changes it without a context.

## tensorplay.backends.openmp

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.backends.openmp.is_available
```

{func}`tensorplay.backends.openmp.is_available` reports whether the OpenMP thread pool
backs CPU parallelism in this build; when it is absent, intra-op parallelism uses the
built-in pool instead.
