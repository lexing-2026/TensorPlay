```{eval-rst}
.. role:: hidden
    :class: hidden-section
```

# tensorplay.library

## Choosing the kind of custom op

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.library.custom_op
    tensorplay.library.triton_op
    tensorplay.library.tile_lang_op
    tensorplay.library.wrap_triton
    tensorplay.library.wrap_tilelang
```

## Extending custom ops (created from Python or C++)

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.library.register_kernel
    tensorplay.library.register_autograd
    tensorplay.library.register_fake
    tensorplay.library.register_vmap
    tensorplay.library.register_autocast
    tensorplay.library.get_kernel
```

## Validation and tooling

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.library.opcheck
    tensorplay.library.infer_schema
```

## Low-level APIs

```{eval-rst}
.. autosummary::
    :toctree: generated
    :nosignatures:

    tensorplay.library.Library
    tensorplay.library.define
    tensorplay.library.impl
    tensorplay.library.impl_abstract
```

## Kernels from raw C++/CUDA via Apache TVM (tvm-ffi)

TVM ships its own Python API (`pip install apache-tvm` / `tvm_ffi`); its
exported functions take any DLPack-compatible tensor as a
`tvm::ffi::TensorView`, and :class:`tensorplay.Tensor` implements the
DLPack protocol — so TP tensors pass through zero-copy with **no adapter**.
Wrap the ffi function in a normal operator to get dispatch, autograd,
fake metadata and the compiler fusion barrier for free:

```python
import tvm_ffi.cpp
import tensorplay as tp
from tensorplay import library

mod = tvm_ffi.cpp.load_inline(          # JIT (cached across runs)
    name="scale",
    cpp_sources=r"""
void scale_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
  for (int64_t i = 0; i < x.size(0); ++i)
    static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] * 3.0f;
}
""",
    functions=["scale_cpu"],
)

@library.custom_op("mylib::triple", mutates_args=())
def triple(x):
    y = tp.empty_like(x)
    mod.scale_cpu(x, y)                 # TP tensors straight through
    return y

@triple.register_fake
def _(x):
    return tp.empty_like(x)

triple.register_autograd(lambda ctx, g: (g * 3.0,))
```

For ahead-of-time builds, replace `load_inline` with
`tvm_ffi.cpp.build_inline(...)` (returns the `.so` path) and load it later
with `tvm_ffi.load_module(path)` on machines without a compiler.
