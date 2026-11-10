<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../../../docs/images/metax-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="../../../docs/images/metax-lockup.svg">
    <img src="../../../docs/images/metax-lockup.svg" alt="MetaX 沐曦" width="420">
  </picture>
</p>

<h1 align="center">tensorplay_metax</h1>

**MetaX (曦云) GPU backend for [TensorPlay]** — the CUDA device backend
compiled for MetaX C500 cards through the MACA SDK (cu-bridge + mxcc).

```python
import tensorplay as tp

x = tp.randn(256, 256, device="cuda")   # a MACA build speaks the CUDA spelling
y = (x @ x).relu()
s = tp.softmax(x, dim=-1)
print(float(y.sum()), float(s.sum()))
```

---

## What this is

TensorPlay's GPU backend is written once, against the CUDA surface, and the
MACA stack is CUDA-compatible enough to target it from the same sources.  In
a `-DUSE_METAX` build:

- device kernels are the **CUDA backend sources compiled unchanged** by
  `mxcc` for the `xcore1000` architecture;
- runtime, BLAS, DNN, random, FFT, solver, sparse and collective calls go
  through the **cu-bridge compatibility headers** (`cuda_runtime.h`,
  `cublas*.h`, `cudnn.h`, `nccl.h`, ...) to the `mc*` libraries;
- the device is spelled `"cuda"` and reached through `tensorplay.cuda`;
- sources that need NVIDIA-only instructions are excluded (see below) and
  their operators take the generic paths;
- the profiler records kernels, copies and fills through `libmcpti`.

Keeping the existing kernels is what makes the performance target
meaningful: the elementwise, reduction, GEMM and convolution paths carry
their CUDA tuning, instead of being rewritten against a vendor library that
covers a fraction of the operators.

### Why this is not a plugin like `tensorplay_mlu`

[`tensorplay_mlu`](https://github.com/ohtensorplay/tensorplay-mlu) targets a
runtime (CNRT/CNNL) whose API has no CUDA counterpart, so its kernels are
written from scratch against the vendor library and it fills TensorPlay's
out-of-tree `PrivateUse1` slot.  MACA instead ships a CUDA-compatible
surface, so the cheaper and more faithful integration is in-tree: the same
source files, selected by the same `USE_CUDA`-family switch as `USE_ROCM`,
with only the build adapted.  That is why this directory holds a toolchain
module and notes rather than a package.

## Requirements

- a MetaX C500 card with the driver loaded (`mx-smi` works)
- MACA SDK (tested with 3.3.0), with `tools/cu-bridge` extracted
- CMake >= 3.18 and a C++20 host compiler

```bash
export MACA_PATH=/opt/maca              # SDK root (default; overrides if relocated)
# export TP_MACA_DEVICE_ARCH=xcore1000  # device arch (default)
```

## Building

```bash
cmake -S . -B build-metax -G Ninja \
  -DUSE_METAX=ON -DCMAKE_BUILD_TYPE=Release
ninja -C build-metax -j 6 p10
ninja -C build-metax -j 6 _C   # the Python bindings; links p10 and the MACA libraries
```

`USE_METAX` and `USE_CUDA` are mutually exclusive.  A MACA build keeps the
CUDA code paths (`USE_CUDA` and `TP_CUDA_ENABLED` are on for the sources),
turns the CUDA language off, and routes every `.cu` through the
`tp_maca_compile_object` rule.  The host-compiled translation units of the
backend (context, allocator, graph, generator) build with the normal C++
compiler against the same headers.

Keep the job count low: device translation units hold several GB while
`mxcc` optimizes them (the elementwise files, with their full dtype fan-out,
peak past 10 GB), and the SDK does not cap its own parallelism.  The device
compiles therefore share one Ninja pool sized from physical memory
(`TP_MACA_MB_PER_JOB`, default 6000); override the pool directly with
`TP_MACA_JOBS=<n>`.

## Compiling one translation unit

The flags below are what `cmake/maca.cmake` produces; they are also the
quickest way to check a source after a header change:

```bash
mxcc -c -std=c++20 -offload-arch=xcore1000 \
  -D__CUDACC__ -Xdevice -D__CUDA_ARCH__=800 -include climits \
  -I p10/include -I p10/src -I p10/src/backend/cuda -I tpx/include \
  -I build/generated -I third_party/cutlass/include \
  -I third_party/cudnn_frontend/include \
  -I /opt/maca/tools/cu-bridge/include -I /opt/maca/include \
  -I /opt/maca/include/mcr -I /opt/maca/include/mcblas \
  -I /opt/maca/include/mcdnn -I /opt/maca/include/mccl \
  -I /opt/maca/include/mcrand -I /opt/maca/include/mcfft \
  -I /opt/maca/include/mcsolver -I /opt/maca/include/mcsparse \
  -DUSE_CUDA -DUSE_CUDNN \
  p10/src/backend/cuda/CopyKernels.cu -o /tmp/copy.o
```

Four of those flags do real work:

- `-D__CUDACC__` — the sources mark host/device helpers with
  `__host__ __device__`, and the compatibility headers only enable those
  annotations in a CUDA compilation. Without it the device side sees host
  functions and fails on the first kernel.
- `-Xdevice -D__CUDA_ARCH__=800` — the device pass selects the
  architecture-specific branches; the host pass must not see the macro.
- `-include climits` — `mc_runtime_api.h` uses `UINT_MAX` without including
  the header that declares it.
- `-DUSE_CUDNN` — the convolution and normalization paths then build against
  the cuDNN-compatible surface (mcdnn behind the compatibility headers);
  keep it so a one-off compile follows the same code paths as the full build.

`build/generated` comes from `tools.codegen.main`; run it once before the
first compile (see the top-level build, which does this automatically).

## Supported operations

The build compiles the whole CUDA backend except what this architecture
cannot run:

| Area | State in a MACA build |
| --- | --- |
| Elementwise, activations, reductions | CUDA kernels compiled by `mxcc` |
| BLAS (`mm`, `addmm`, `bmm`, ...) | CUDA kernels calling `mcblas` through the cuBLAS compatibility headers |
| Convolution | the shared and v7 cuDNN paths build against `mcdnn`; the v8 cuDNN-frontend graph API is excluded |
| Quantized / grouped GEMM | excluded — inline PTX (`mma.sync`, `lop3`, `cp.async`); operators fall back to the generic paths |
| Transformers attention | excluded — the flash and wide-tiled schedules draw on the same tensor-core surface; falls back |
| Profiling | per-operation device time, kernel-level records through `libmcpti` |

## Profiling

```python
from tensorplay.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step()
print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
```

The MACA tracing library keeps the CUPTI entry points under its own name
(`libmcpti.so`), so the profiler loads it first in a MACA build and the
records arrive through the same path as on NVIDIA.

## Layout

| Path | Role |
| --- | --- |
| `cmake/maca.cmake` | SDK discovery, the `mxcc` compile rule, the device job pool, host/device include separation |
| `bench_ops.py` | operator benchmark against the reference library, one library per process |
| root `CMakeLists.txt` | the `USE_METAX` switch, SDK discovery, `_C` linkage |
| `p10/CMakeLists.txt` | source selection, per-TU `mxcc` objects, the exclusion list |
| `p10/src/backend/cuda/*.cu` | the reused device kernels |
| `p10/include/BFloat16.h`, `p10/include/Half.h` | conversion intrinsics take integer paths under `mxcc` |
| `p10/src/ProfilerCupti.cpp` | `libmcpti.so` as the tracing library in a MACA build |

Paths without a prefix are relative to the repository root; the two files
under this directory live in `backends/metax/tensorplay_metax/`.

## Status

Verified on a MetaX C500 (64 GB, MACA 3.3.0):

- a minimal `cudaMalloc` / kernel / `cudaMemcpy` program compiles and runs;
- `CopyKernels.cu`, `Convolution.cu`, `TensorFactories.cu` and
  `ActivationKernels.cu` compile with `mxcc`.

## Compatibility

| Component | Version |
| --- | --- |
| TensorPlay | in-tree backend; moves with the framework (`main`) |
| MACA SDK | 3.3.0 (`/opt/maca`) |
| Device | MetaX C500 (`xcore1000`) |
| `mxcc` | cu-bridge, LLVM/clang 19 based |

## License

Apache License 2.0, as the rest of the repository.

*MetaX, 曦云 and 沐曦 are trademarks of MetaX Technologies Inc.  This project
is not affiliated with or endorsed by MetaX Technologies.  The lockup above
is the vendor's public logo, used to identify the supported hardware.*

[TensorPlay]: https://github.com/lexing-2026/TensorPlay
