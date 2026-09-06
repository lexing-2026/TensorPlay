(reproducibility)=

# Reproducibility

Completely reproducible results are not guaranteed across TensorPlay releases,
individual commits, or different platforms. Furthermore, results may not be
reproducible between CPU and GPU executions, even when using identical seeds.

However, there are some steps you can take to limit the number of sources of
nondeterministic behavior for a specific platform, device, and TensorPlay release.
First, you can control sources of randomness that can cause multiple executions
of your application to behave differently. Second, you can configure TensorPlay
to avoid using nondeterministic algorithms for some operations, so that multiple
calls to those operations, given the same inputs, will produce the same result.

:::{warning}
Deterministic operations are often slower than nondeterministic operations, so
single-run performance may decrease for your model. However, determinism may
save time in development by facilitating experimentation, debugging, and
regression testing.
:::

## Controlling sources of randomness

### TensorPlay random number generator

You can use {meth}`tensorplay.manual_seed()` to seed the RNG for all devices (both
CPU and CUDA):

```python
import tensorplay
tensorplay.manual_seed(0)
```

Some TensorPlay operations may use random numbers internally.
{meth}`tensorplay.svd_lowrank()` does this, for instance. Consequently, calling it
multiple times back-to-back with the same input arguments may give different
results. However, as long as {meth}`tensorplay.manual_seed()` is set to a constant
at the beginning of an application and all other sources of nondeterminism have
been eliminated, the same series of random numbers will be generated each time
the application is run in the same environment.

It is also possible to obtain identical results from an operation that uses
random numbers by setting {meth}`tensorplay.manual_seed()` to the same value between
subsequent calls.

### Python

For custom operators, you might need to set python seed as well:

```python
import random
random.seed(0)
```

### Random number generators in other libraries

If you or any of the libraries you are using rely on NumPy, you can seed the global
NumPy RNG with:

```python
import numpy as np
np.random.seed(0)
```

However, some applications and libraries may use NumPy Random Generator objects,
not the global RNG (<https://numpy.org/doc/stable/reference/random/generator.html>),
and those will need to be seeded consistently as well.

If you are using any other libraries that use random number generators, refer to
the documentation for those libraries to see how to set consistent seeds for them.

### CUDA convolution benchmarking

The cuDNN library, used by CUDA convolution operations, can be a source of nondeterminism
across multiple executions of an application. When a cuDNN convolution is called with a
new set of size parameters, an optional feature can run multiple convolution algorithms,
benchmarking them to find the fastest one. Then, the fastest algorithm will be used
consistently during the rest of the process for the corresponding set of size parameters.
Due to benchmarking noise and different hardware, the benchmark may select different
algorithms on subsequent runs, even on the same machine.

Disabling the benchmarking feature with `tensorplay.backends.cudnn.benchmark = False`
causes cuDNN to deterministically select an algorithm, possibly at the cost of reduced
performance.

However, if you do not need reproducibility across multiple executions of your application,
then performance might improve if the benchmarking feature is enabled with
`tensorplay.backends.cudnn.benchmark = True`.

Note that this setting is different from the `tensorplay.backends.cudnn.deterministic`
setting discussed below.

## Avoiding nondeterministic algorithms

{meth}`tensorplay.use_deterministic_algorithms` lets you configure TensorPlay to use
deterministic algorithms instead of nondeterministic ones where available, and
to throw an error if an operation is known to be nondeterministic (and without
a deterministic alternative).

Please check the documentation for {meth}`tensorplay.use_deterministic_algorithms()`
for a full list of affected operations. If an operation does not act correctly
according to the documentation, or if you need a deterministic implementation
of an operation that does not have one, please submit an issue:
<https://github.com/tensorplay/tensorplay/issues?q=label:%22module:%20determinism%22>

For example, running the nondeterministic CUDA implementation of {meth}`tensorplay.Tensor.index_add_`
will throw an error:

```python
>>> import tensorplay
>>> tensorplay.use_deterministic_algorithms(True)
>>> tensorplay.randn(2, 2).cuda().index_add_(0, tensorplay.tensor([0, 1]), tensorplay.randn(2, 2))
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
RuntimeError: index_add_cuda_ does not have a deterministic implementation, but you set
'tensorplay.use_deterministic_algorithms(True)'. ...
```

When {meth}`tensorplay.bmm` is called with sparse-dense CUDA tensors it typically uses a
nondeterministic algorithm, but when the deterministic flag is turned on, its alternate
deterministic implementation will be used:

```python
>>> import tensorplay
>>> tensorplay.use_deterministic_algorithms(True)
>>> tensorplay.bmm(tensorplay.randn(2, 2, 2).to_sparse().cuda(), tensorplay.randn(2, 2, 2).cuda())
tensor([[[ 1.1900, -2.3409],
         [ 0.4796,  0.8003]],
        [[ 0.1509,  1.8027],
         [ 0.0333, -1.1444]]], device='cuda:0')
```

### CUDA convolution determinism

While disabling CUDA convolution benchmarking (discussed above) ensures that
CUDA selects the same algorithm each time an application is run, that algorithm
itself may be nondeterministic, unless either
`tensorplay.use_deterministic_algorithms(True)` or
`tensorplay.backends.cudnn.deterministic = True` is set. The latter setting
controls only this behavior, unlike {meth}`tensorplay.use_deterministic_algorithms`
which will make other TensorPlay operations behave deterministically, too.

### CUDA Scaled Dot Product Attention

{func}`tensorplay.nn.functional.scaled_dot_product_attention` (SDPA) dispatches to
multiple backends at runtime. Each backend has different determinism
characteristics, summarized in the table below:

```{list-table}
:header-rows: 1
:widths: 25 15 15 45

* - Backend
  - Forward
  - Backward
  - Notes
* - `SDPBackend.MATH`
  - Deterministic
  - Deterministic
  - Uses standard TensorPlay operators (`matmul`, `softmax`). Deterministic
    when {meth}`tensorplay.use_deterministic_algorithms` is enabled.
* - `SDPBackend.FLASH_ATTENTION`
  - Deterministic
  - Non-deterministic
  - The backward pass uses non-deterministic atomic operations by default.
    Setting `tensorplay.use_deterministic_algorithms(True, warn_only=False)`
    enables a deterministic backward implementation.
* - `SDPBackend.EFFICIENT_ATTENTION`
  - Deterministic
  - Non-deterministic
  - The backward pass may split work across keys (`num_splits_key > 1`)
    for performance, which is non-deterministic. Setting
    `tensorplay.use_deterministic_algorithms(True, warn_only=False)` forces
    `num_splits_key = 1`, making the backward pass deterministic.
* - `SDPBackend.CUDNN_ATTENTION`
  - Deterministic
  - Non-deterministic
  - cuDNN provides an opt-in deterministic backward for `float16`
    starting in
    [cuDNN 9.18](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/release-notes.html#cudnn-9-18-0)
    (via
    [set_deterministic_algorithm](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/operations/Attention.html#sdpa-fp16-bf16-backward)),
    but this has not yet been integrated into TensorPlay. This backend is
    **disabled** when
    `tensorplay.use_deterministic_algorithms(True, warn_only=False)` is set,
    regardless of whether inputs require gradients or the call is
    inside a `tensorplay.no_grad()` context.
```

When `tensorplay.use_deterministic_algorithms(True, warn_only=True)` is set,
the fused backends (Flash, Efficient, and cuDNN) emit a one-time warning but
still use their default non-deterministic code paths. To actually enforce
determinism, pass `warn_only=False`.

**Low-precision dtypes and numerical reproducibility**

Bitwise matching numerics across different SDPA backends are **not
guaranteed**, even for the same inputs and dtype. Each backend performs
floating-point accumulation in a different order, and because floating-point
addition is not associative, the results will differ between backends.

The Math backend by default accumulates in `float32` even when given
`float16` or `bfloat16` inputs. The function
{func}`tensorplay.backends.cuda.allow_fp16_bf16_reduction_math_sdp` can enable
reduced-precision accumulation for higher performance at the cost of
different numerical results.

**Selecting a specific backend**

Use the {func}`tensorplay.nn.attention.sdpa_kernel` context manager to restrict
which backends SDPA may use. For example, to guarantee deterministic behavior
regardless of hardware:

```python
import tensorplay
from tensorplay.nn.attention import sdpa_kernel, SDPBackend

tensorplay.use_deterministic_algorithms(True)

# Option 1: Use only the Math backend (always deterministic)
with sdpa_kernel(SDPBackend.MATH):
    out = tensorplay.nn.functional.scaled_dot_product_attention(q, k, v)

# Option 2: Allow Flash and Efficient (deterministic with the flag above)
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    out = tensorplay.nn.functional.scaled_dot_product_attention(q, k, v)
```

### CUDA RNN and LSTM

In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
See {meth}`tensorplay.nn.RNN` and {meth}`tensorplay.nn.LSTM` for details and workarounds.

### Filling uninitialized memory

Operations like {meth}`tensorplay.empty` and {meth}`tensorplay.Tensor.resize_` can return
tensors with uninitialized memory that contain undefined values. Using such a
tensor as an input to another operation is invalid if determinism is required,
because the output will be nondeterministic. But there is nothing to actually
prevent such invalid code from being run. So for safety,
{attr}`tensorplay.utils.deterministic.fill_uninitialized_memory` is set to `True`
by default, which will fill the uninitialized memory with a known value if
`tensorplay.use_deterministic_algorithms(True)` is set. This will prevent the
possibility of this kind of nondeterministic behavior.

However, filling uninitialized memory is detrimental to performance. So if your
program is valid and does not use uninitialized memory as the input to an
operation, then this setting can be turned off for better performance.

## DataLoader

DataLoader will reseed workers following the {ref}`data-loading-randomness` algorithm.
Use {meth}`worker_init_fn` and `generator` to preserve reproducibility:

```python
def seed_worker(worker_id):
    worker_seed = tensorplay.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

g = tensorplay.Generator()
g.manual_seed(0)

DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    worker_init_fn=seed_worker,
    generator=g,
)
```
