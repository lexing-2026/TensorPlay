(new-backends)=
# Adding a device backend

TensorPlay reserves one device slot, `PrivateUse1`, for a backend that lives
outside this repository: a new accelerator, a simulator, a remote device. The
slot is a full backend. It has its own dispatch key and its own autograd,
autocast and vmap layers, and it can take a name of its own, so
`tp.ones(3, device="npu")` reads like any other device.

The reference backend under `test/cpp_extensions/openreg/` is about 200 lines
and implements everything below. Read it alongside this note.

## What a backend provides

| Piece | Where it is registered | What it answers |
| --- | --- | --- |
| Allocator | `REGISTER_ALLOCATOR(DeviceType::PrivateUse1, &a)` (`Allocator.h`) | where every tensor on the device gets its memory |
| Device hooks | `REGISTER_PRIVATEUSE1_HOOKS(&h)` (`PrivateUse1Hooks.h`) | device count, current device, synchronization, raw byte copies |
| Kernels | `TENSORPLAY_LIBRARY_IMPL(PrivateUse1, Name)` (`Dispatcher.h`) | the operators themselves |
| Name | `tensorplay.utils.rename_privateuse1_backend("npu")` | how the device is spelled in Python |

The C++ pieces register from static initializers, so loading the library is
all it takes. `tensorplay.utils.cpp_extension.load` compiles sources against
the `p10` headers, links `libp10`, caches the build and loads it:

```python
import tensorplay as tp
from tensorplay.utils import cpp_extension

cpp_extension.load("npu_backend", ["npu_backend.cpp"])
tp.utils.rename_privateuse1_backend("npu")

x = tp.randn(4, 4).to("npu")
```

## Kernels

Every backend implements four operators natively, because everything else is
built on them:

- `empty` and `empty.memory_format` allocate from the backend's allocator;
- `copy_` moves data between the device and the host in both directions,
  and between two devices of the backend;
- `item` reads a single value back to the host.

Views (`view`, `as_strided`, `t`, slicing, ...) are composite kernels: they
only rewrite sizes and strides, so they serve the new device with no work.
So do composite operators built from other operators (`_to_copy`,
`empty_strided`, `ones`, ...), once the operators they call have kernels.

A kernel is called through a pointer cast to the operator's kernel
signature, so a kernel of a different type is undefined behaviour.
`tensorplay/ops/KernelSignatures.h` spells the signature of every operator
as a C++ function type. Casting to it makes a mismatch a compile error:

```cpp
#include "tensorplay/ops/KernelSignatures.h"

namespace sig = tensorplay::kernel_signatures;

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NpuKernels) {
    m.impl("empty", static_cast<sig::empty*>(&npu_empty));
    m.impl("copy_", static_cast<sig::copy_*>(&npu_copy));
    m.impl("mm", static_cast<sig::mm*>(&npu_mm));
}
```

Names replace the overload dot with two underscores: `add.Tensor` becomes
`sig::add__Tensor`.

### Bringing a model up with host fallbacks

A backend rarely starts with a kernel for every operator. `CpuFallback.h`
turns any operator's CPU kernel into a kernel for the device. It copies the
device arguments to the host, runs the CPU kernel, then copies the results
and the written arguments back:

```cpp
#include "CpuFallback.h"

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NpuHostFallbacks) {
    m.impl("addmm", &cpu_fallback::kernel<"addmm", sig::addmm>);
    m.impl("add_.Tensor", &cpu_fallback::kernel<"add_.Tensor", sig::add___Tensor>);
}
```

The fallback moves data only with the backend's own `empty` and `copy_`, so
it works on hardware the host cannot address. Every call pays two transfers
per tensor. Use it to get a model running, then replace fallbacks with
device kernels one operator at a time. It cannot serve operators whose
result aliases an input (views); the composite view kernels already cover
those.

An operator with no kernel for the device fails loudly with `Kernel not
found for op: <name> on backend: PrivateUse1`. Set
`TP_TRACE_KERNEL_MISS=1` to also print the call stack that reached it.

### Autograd, autocast and vmap come with the slot

Derivative formulas are written in terms of operators, and the generated
autograd kernels are registered for `AutogradPrivateUse1` like for every
in-tree backend. Once the forward and backward operators a model uses have
kernels, `loss.backward()` works on the new device. Autocast uses the CUDA
policy (float16 by default) under `tp.autocast("npu")`. `tp.func.vmap`
batches device tensors through the same batch rules as the host.

## Device hooks

The hooks object answers device-level questions the runtime cannot answer
by itself. Every method has a default that fits one synchronous device, so
a backend overrides only what differs:

```cpp
struct NpuHooks final : tensorplay::PrivateUse1HooksInterface {
    int64_t device_count() const override { return npu_device_count(); }
    int64_t current_device() const override { return npu_get_device(); }
    void set_device(int64_t i) const override { npu_set_device(i); }
    void synchronize(int64_t i) const override { npu_synchronize(i); }
    void copy_bytes(void* dst, const tensorplay::Device&, const void* src,
                    const tensorplay::Device&, size_t n) const override {
        npu_memcpy(dst, src, n);
    }
};

NpuHooks hooks;
REGISTER_PRIVATEUSE1_HOOKS(&hooks);
```

A named backend with hooks becomes the process's accelerator, ahead of any
in-tree one. `tp.accelerator.current_accelerator()`, `device_count()`,
`set_device_index()`, `device_index(...)` and `synchronize()` then answer
for it.

## Python side

- `tp.utils.rename_privateuse1_backend(name)` names the device. The name is
  fixed for the process. It must be an identifier and may not be an in-tree
  device type. `"privateuseone"` keeps parsing afterwards.
- `tp.utils.generate_methods_for_privateuse1_backend()` adds
  `Tensor.npu()`, `Tensor.is_npu`, `Module.npu()` and the `PackedSequence`
  counterparts.
- `tp._register_device_module("npu", module)` installs a runtime module as
  `tp.npu`. Without one, `tp.npu` is a default module built on the hooks:
  `is_available`, `device_count`, `current_device`, `set_device`,
  `synchronize`.
- Custom operators take device kernels under the slot's names:
  `op.register_kernel("PrivateUse1")` (or `"privateuseone"`, or the
  registered name) serves tensors on the device. The registration can come
  before or after the rename.

## Packaging

A backend shipped as a wheel declares an entry point in the
`tensorplay.backends` group. `import tensorplay` calls every such entry
point, so the backend loads its library and names its device without the
user importing anything:

```toml
[project.entry-points."tensorplay.backends"]
npu = "tensorplay_npu:_autoload"
```

Set `TENSORPLAY_DEVICE_BACKEND_AUTOLOAD=0` to skip autoloading.
