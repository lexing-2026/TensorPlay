# tensorplay_metax

Device backend for MetaX (曦云) GPUs, built on the MACA SDK.

## Why this backend does not look like `tensorplay_mlu`

`tensorplay_mlu` targets a runtime (CNRT/CNNL) whose API has no CUDA
counterpart, so its kernels are written from scratch against the vendor
library and the rest of the operator surface goes through host fallbacks.

MACA is different: `tools/cu-bridge` ships a CUDA-compatible surface
(`cuda_runtime.h`, `cublas*.h`, `cudnn.h`, `nccl.h`, `curand*.h`, `cufft.h`,
`cusolver*.h`, `cusparse*.h`, `cub`, `thrust`, `cutlass`) whose wrapper
headers route each call to the `mc*` libraries, and `mxcc` compiles CUDA
translation units for the `xcore1000` architecture. The CUDA backend sources
therefore compile unchanged; only the build has to select `mxcc` and the
MACA libraries.

Keeping the existing kernels is what makes the performance target
meaningful: the elementwise, reduction, GEMM and convolution paths carry
their CUDA tuning, instead of being rewritten against a vendor library that
covers a fraction of the operators.

## Layout

```
backends/metax/tensorplay_metax/
├── cmake/maca.cmake     locate the SDK, compile a .cu with mxcc
├── bench_ops.py         operator benchmark against the reference library
└── README.md
```

The build itself is selected from the top-level tree with `-DUSE_METAX=ON`;
this directory only carries the toolchain module and the notes.

## Requirements

- MACA SDK (tested with 3.3.0 at `/opt/maca`), driver loaded (`mx-smi` works)
- cu-bridge extracted to `/opt/maca/tools/cu-bridge`
- CMake >= 3.18, a C++20 host compiler

`MACA_PATH` overrides the SDK root; `TP_MACA_DEVICE_ARCH` overrides
`xcore1000`.

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

## Status

Verified on a MetaX C500 (64 GB, MACA 3.3.0):

- a minimal `cudaMalloc` / kernel / `cudaMemcpy` program compiles and runs;
- `CopyKernels.cu`, `Convolution.cu`, `TensorFactories.cu` and
  `ActivationKernels.cu` compile with `mxcc`.

Build integration and runtime bring-up are in progress.
