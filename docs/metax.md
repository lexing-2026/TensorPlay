<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="images/metax-lockup-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="images/metax-lockup.svg">
    <img src="images/metax-lockup.svg" alt="MetaX 沐曦" width="420">
  </picture>
</p>

<h1 align="center">MetaX (MACA) backend</h1>

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
with only the build adapted.  MetaX therefore needs no package of its own —
a toolchain module (`cmake/LoadMACA.cmake`) plus the reuse of the existing
CUDA sources.

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

The runtime-compilation pair (`JitUtils.cpp`, `Jiterator.cpp`) is host-side
code, yet one of its headers reaches the device vector-type definitions,
which only the SDK's own driver exposes on its search path.  Both files
therefore go through `mxcc` as well, in a **host mode** that drops the device
architecture and the device-language macro — so the compatibility headers
keep the host definitions of the standard library — and takes a narrower
include set that leaves out the template-library directories.

Keep the job count low: device translation units hold several GB while
`mxcc` optimizes them (the elementwise files, with their full dtype fan-out,
peak past 10 GB), and the SDK does not cap its own parallelism.  The device
compiles therefore share one Ninja pool sized from physical memory
(`TP_MACA_MB_PER_JOB`, default 6000); override the pool directly with
`TP_MACA_JOBS=<n>`.

## Publishing a wheel

There is no CI lane: the MACA SDK is not publicly downloadable and the
acceptance run needs a real card, so a MetaX wheel is built by hand on a
C500 and attached to a GitHub pre-release.  Reuse the build tree from the
sections above so the wheel does not recompile the device image:

```bash
export MACA_PATH=/opt/maca PATH=/opt/maca/mxgpu_llvm/bin:$PATH
export TENSORPLAY_BUILD_VERSION=1.3.0a0+maca330
export SKBUILD_BUILD_DIR=build-metax
export SKBUILD_CMAKE_DEFINE="USE_CUDA=OFF;USE_METAX=ON;USE_MPI=OFF"
python -m build --wheel --no-isolation --outdir dist
```

The `+maca330` local label keeps the wheel off the `cuXXX` and `rocmXXX`
indexes; `tools/generate_tensorplay_version.py` only knows the CUDA, ROCm
and CPU variants, so the label is passed in whole.  Upload with
`gh release upload <tag> dist/*.whl`; the release notes must state the
MACA SDK version and that the SDK is a runtime requirement, not a bundled
one.

Verify the artifact before uploading: install it into a throwaway target
directory and run `benchmark/validate_metax_backend.py --device cuda`,
which reports a verdict per case and exits non-zero on any failure.

## Compiling one translation unit

The flags below are what `cmake/LoadMACA.cmake` produces; they are also the
quickest way to check a source after a header change:

```bash
mxcc -c -std=c++20 -offload-arch=xcore1000 \
  -D__CUDACC__ -Xdevice -D__CUDA_ARCH__=800 -include climits \
  -I p10/include -I p10/src -I p10/src/backend/cuda -I tpx/include \
  -I build/generated -I build-metax/metax_cutlass_shim \
  -I /opt/maca/tools/cu-bridge/include -I /opt/maca/include \
  -I /opt/maca/include/mcr -I /opt/maca/include/mcblas \
  -I /opt/maca/include/mcdnn -I /opt/maca/include/mccl \
  -I /opt/maca/include/mcrand -I /opt/maca/include/mcfft \
  -I /opt/maca/include/mcsolver -I /opt/maca/include/mcsparse \
  -I /opt/maca/include/mctlass -I /opt/maca/include/mccub \
  -I /opt/maca/include/cute \
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
`metax_cutlass_shim` is a forward tree the configure step writes so that a
`<cutlass/...>` include resolves to the SDK's template headers; both
directories exist once the build has been configured.

Host-side files that still include the SDK headers — the runtime-compilation
pair — go through the same driver with the device options removed:

```bash
mxcc -c -std=c++20 -DUSE_CUDA -DUSE_CUDNN -DUSE_METAX -include climits \
  -fPIC -O3 \
  -I p10/include -I p10/src -I p10/src/backend/cuda -I tpx/include \
  -I build/generated \
  -I /opt/maca/tools/cu-bridge/include -I /opt/maca/include \
  -I /opt/maca/include/common -I /opt/maca/include/mcr \
  p10/src/backend/cuda/JitUtils.cpp -o /tmp/jitutils.o
```

No `-offload-arch`, no `-D__CUDACC__`, no `-Xdevice`: the translation unit has
no device pass, and the compatibility headers must keep the host definitions
of the standard library. The template-library directories stay off the line,
while `include/common` joins it to supply the fp16 and complex headers the
compatibility surface reaches for.

## Supported operations

The build compiles the whole CUDA backend except what this architecture
cannot run:

| Area | State in a MACA build |
| --- | --- |
| Elementwise, activations, reductions | CUDA kernels compiled by `mxcc` |
| BLAS (`mm`, `addmm`, `bmm`, ...) | CUDA kernels calling `mcblas` through the cuBLAS compatibility headers |
| Convolution | the shared, v7 and v8 kernel files build against `mcdnn`; the v8 graph-plan frontend compiles out, so `conv2d` runs the legacy path and the gradient kernels use the descriptor API |
| Quantized GEMM | excluded — inline PTX (`mma.sync`, `lop3`, `cp.async`); operators fall back to the generic paths |
| Grouped GEMM | excluded — the grouped tensor-core schedule is not shipped; a fallback entry point reports every configuration as outside its envelope, so callers keep the per-group math-library path |
| Transformers attention | the flash and wide-tiled schedules are excluded; `scaled_dot_product_attention`, its backward and the log-sum-exp variants register the composed schedule instead, so attention, its backward and `TransformerEncoderLayer` run |
| Sparse linear algebra | the solver library builds, but the sparse-solver handle cannot be created on this device, so the sparse solve entry point raises when reached |
| CUDA Graph capture and replay | capture, instantiate and replay run against the compatibility runtime's graph entry points; conditional nodes (CUDA >= 12.4) are excluded by the version gate and refused at the API |
| Profiling | per-operation device time, kernel-level records through `libmcpti` |

## Graph compilation

`stax`, the default backend of `tp.compile`, lowers a captured graph through the
kernel-writing runtime MetaX ships as its own fork of Triton.  No other part of
the framework needs it, and this backend imports it only while a graph is being
compiled: with it absent, the device is driven exactly as before through the
kernel library, and the backend reports that there is no runtime rather than
lowering anything.

Installing that fork is what turns graph compilation on:

```bash
pip install --break-system-packages --no-deps \
  "triton==3.0.0+metax3.3.0.2" \
  -i https://repos.metax-tech.com/r/maca-pypi/simple \
  --trusted-host repos.metax-tech.com
```

The version has to be the one built for the installed MACA release (3.3.0 pairs
with this one); it is a wheel, so no compiler toolchain is needed to install it,
and it is not needed for anything else.

A device driven by this stack is named for the stack inside the kernel-writing
runtime (`maca`, 64-wide warps) and is compiled with the warp width and the
options that stack's backend defines.  The framework still presents the same
device as an accelerator, so the launch path translates the name where it looks
a device up, and a precompiled kernel is read from the stack's own image name
(`.mcfatbin`).

Measured on the C500 above, warm, against eager and against the reference
framework's own graph compiler running the same function:

| Case | eager | `stax` | reference compiler |
| --- | --- | --- | --- |
| `(a+b).relu()*2`, 4M elements | 113 us | 50 us (2.3x) | 47 us (2.2x) |
| `(a+b).relu()*2`, 16M elements | 361 us | 144 us (2.5x) | 194 us (1.7x) |
| `(a+b).relu()*2`, 64M elements | 1352 us | 538 us (2.5x) | 751 us (1.7x) |
| `softmax`, 4096x4096 | 440 us | 240 us (1.8x) |  |
| `softmax`, 8192x8192 | 1769 us | 1170 us (1.5x) |  |
| `softmax`, 65536x256 | 653 us | 151 us (4.3x) |  |
| `matmul`, 4096x4096 | 4.70 ms | 4.70 ms |  |

Every case agrees with eager: the elementwise chain and the matmul exactly, the
softmax to 6e-8.  The matmul is unchanged because a matrix multiply is taken as
a library call rather than a generated kernel, so the gain is in the fused
elementwise and reduction work around it.  A chain that becomes one kernel still
pays a fixed cost of about 40 us per call, which is more than the kernel itself
costs below a few million elements -- that is why the 4M row is where `stax` and
the reference compiler meet rather than pull apart.

`softmax` and its backward are handed to purpose-written row kernels where the
device language those need is installed; where it is not, the call keeps the
ordinary reduction lowering, which is what runs here, so the operation still
compiles rather than being given up on.

## Convolution

`conv2d` runs on the legacy descriptor API.  mcDNN serves the descriptor-style
cuDNN surface, but the framework's convolution graph plan is compiled out for
this backend (see `p10/CMakeLists.txt`), which closes the way to the
tensor-core engines the reference stack uses.  Through the descriptor surface
the forward algorithm is not tuned:
`cudnnGetConvolutionForwardAlgorithm_v7` and
`cudnnFindConvolutionForwardAlgorithmEx` both return only an implicit
precomputed GEMM and FFT tiling, each with a unit estimate and no measured
time, and forcing any other algorithm identifier yields the same kernel.  On a
3x3 convolution with 32 or more input channels that costs 1.3x-1.7x against
the reference, which reaches mcDNN's channel-last TF32 tensor-core engines
through the graph API; a 3-channel input convolution is faster here, and the
gradient kernels on the same shapes match or beat the reference.  The gap
therefore sits inside a training step that is also launch-bound rather than in
one operator.

Asking for the operands in channel-last order does not change the selection:
the same descriptors with channel-last strides pick the same engine, with
identical numerics, and only add the repack.  FFT tiling is 1.25x-2.6x faster
on the affected forward shapes, but it doubles the float32 error and gives no
end-to-end gain in the training step, so it is not selected.

## Workload benchmarks

One library per process on the same C500, identical weights and inputs, medians
over repeated runs; the ratio is TensorPlay / reference, so below one is
faster.

| Workload | TensorPlay | reference | ratio |
| --- | --- | --- | --- |
| VL decode, text prompt, bf16 | 16.2 ms/token | 33.4 ms/token | 0.49 |
| VL decode, image prompt, bf16 | 16.2 ms/token | 29.0 ms/token | 0.56 |
| VL prefill, text / image | 0.029 / 0.064 s | 0.035 / 0.067 s | 0.83 / 0.96 |
| LLM prefill, 128 / 512 / 2048 tokens | 2.53 / 6.20 / 30.46 ms | 3.02 / 7.13 / 31.04 ms | 0.84 / 0.87 / 0.98 |
| LLM decode, 128/1 / 512/4 / 2048/4 | 2.53 / 8.43 / 54.08 ms/token | 3.05 / 7.76 / 42.70 ms/token | 0.83 / 1.09 / 1.27 |

The VL figures decode one token per step after a single prefill, so prefill and
per-token cost are read separately.  The LLM figures are
`benchmark/bench_llama_e2e.py`, bf16 without a KV cache; the decode ratio rises
with sequence length, where the matmul that the tensor-core gap above also
affects dominates the step.

`softmax` over 4096 rows is 1.04x, 1.12x and 1.17x the reference in float32 at
widths 2048, 4096 and 8192; 1.62x, 1.39x, 1.24x and 1.21x in float16 at widths
2048 through 16384; and 1.60x, 1.48x, 1.35x and 1.56x in bfloat16 at the same
widths.  At width 16384 in float32 it is faster than the reference (0.60x).
Narrow and 16-bit rows are launch- and bandwidth-bound, which is where the
deficit remains.

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
| `cmake/LoadMACA.cmake` | SDK discovery, the `mxcc` compile rule, the device job pool, host/device include separation |
| `cmake/mc_fatbin_layout.ld` | places the collected device image (`.mc_fatbin`) after the writable segment, so the host code's relocations stay in range |
| `benchmark/bench_metax_ops.py` | operator benchmark against the reference library, one library per process |
| root `CMakeLists.txt` | the `USE_METAX` switch, SDK discovery, `_C` linkage |
| `p10/CMakeLists.txt` | source selection, per-TU `mxcc` objects, the exclusion list |
| `p10/src/backend/cuda/*.cu` | the reused device kernels |
| `p10/src/backend/cuda/GroupedGemmFallback.cpp` | the grouped-GEMM entry point the excluded kernel file would have defined |
| `p10/src/backend/cuda/transformers/SdpaComposed.cpp` | the attention entry points the excluded kernel file would have defined, answered by the composed schedule |
| `benchmark/model_smoke.py` | LLM and CV training steps against the reference library, same weights in both runs |
| `p10/include/BFloat16.h`, `p10/include/Half.h` | conversion intrinsics take integer paths under `mxcc` |
| `p10/src/ProfilerCupti.cpp` | `libmcpti.so` as the tracing library in a MACA build |

All paths are relative to the repository root.

## Status

Verified on a MetaX C500 (64 GB, MACA 3.3.0), Python 3.12:

- `p10` and the `_C` extension build cleanly, and `libp10.so` links with no
  unresolved backend symbols.  The collected device image is large enough to
  push `.got`/`.tbss` beyond the reach of the host code's 32-bit relocations,
  so it is placed at the end of the image (`cmake/mc_fatbin_layout.ld`); the
  image is only reached through an absolute pointer, so its position is
  immaterial to the runtime;
- `import tensorplay` succeeds and the runtime reports a single device,
  `MetaX C500`;
- a BLAS matmul, `softmax` and elementwise arithmetic run and produce finite
  results;
- the reduction path matches the reference library: `sum(1d)` over 127..16M
  elements lands within 0.91x-1.07x of the reference latency (0.69x-1.39x
  across the whole reduction bench), and a 1024-element sum costs about 10 us
  per launch;
- the reduction regression suites pass on the device (77 passed, 3 skipped);
- broadcast binary arithmetic (`add`/`sub`/`mul`/`div`) uses row kernels instead
  of a per-element index decomposition: a trailing-dimension broadcast
  (`(2,8,512,512) - (2,8,512,1)`) runs at about 890 GB/s against 1380 GB/s for
  the same shape, and a channel broadcast `(N,C,H,W) op (N|1,C,1,1)` reaches
  about 1.2 TB/s.  A broadcast along a middle dimension
  (`(2,8,512,512) - (2,1,512,512)`) still walks one coordinate per row and stays
  the slowest broadcast shape at about 250 GB/s;
- sorting matches the reference along every dimension: `sort`/`argsort` agree
  with the CPU over 2-D and 3-D shapes, every axis, float32/float64/int32/int64
  and both orders, and `test_sort_native.py` passes (24 passed); the backward
  that consumes the permutation (`msort` / `take_along_dim`) passes too;
- top-k matches the CPU over rows from 2 to 25000 wide, one-dimensional and
  batched, both directions and with `sorted` set either way; the selection
  histogram is counted with hardware-width ballots, so the counts stay exact
  when a hardware warp spans more than one 32-thread group (1120 randomized
  checks, `test_topk_native.py` 33 passed);
- a CV training step (three convolutions with batch norm, max pooling, a
  classifier head; 8x3x32x32) and an LLM training step (two transformer
  blocks, `d=256`, 8 heads, sequence 128, batch 4, `scaled_dot_product_attention`)
  both run forward, backward and the update.  Fed identical weights and
  inputs, the two libraries agree: the loss trajectory matches to six
  decimals, the matmul gradients to 2e-7 relative, and the largest
  per-parameter gradient deviation is 1.6e-3 (CV) and 5.1e-3 (LLM), both on
  small normalization-scale gradients where the reduction order differs;
- the same steps cost 2.20 ms against 1.72 ms (CV), 8.78 ms against 6.46 ms
  (ResNet-18, 11M parameters) and 12.83 ms against 8.19 ms (four transformer
  blocks, `d=512`, sequence 512), a 1.24x-1.57x gap that widens with sequence
  length (`benchmark/model_smoke.py`);
- the profiler loads `libmcpti.so` and records per-operation device time and
  kernel-level entries;
- CUDA Graph capture, instantiate and replay round-trip a matmul node: the
  replayed result matches the value computed outside the graph;
- conditional nodes are refused — the compatibility runtime reports CUDA
  11.6, below the 12.4 the gate requires;
- the sparse-solver handle cannot be created on this device, so sparse solves
  raise when reached (the operation table records the exclusion).

The full `test/` suite is not yet a clean signal here: on the reference box it
is dominated by a NumPy 2 ABI mismatch and by a sticky illegal-memory-access
cascade, so its failure count does not measure backend coverage.  The earlier
`msort` / `take_along_dim` trap was a consequence of the sort kernel rather
than of the scatter it appeared in: `sort` along a dimension other than the
last returned zeroed and shuffled rows, and the out-of-range permutation it
produced then drove the backward scatter past the end of its output buffer.

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
