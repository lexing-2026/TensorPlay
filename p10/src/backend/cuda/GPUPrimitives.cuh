#pragma once

// Single include point for the GPU collectives (block/warp/device-level
// primitives) used by the kernel backends.
//
// The two GPU toolchains ship these primitives under different names: the
// CUDA toolchain provides the CUB headers and the `cub` namespace, while
// the HIP toolchain provides hipCUB with the `hipcub` namespace over the
// rocPRIM implementation.  The class templates, enum spellings and call
// shapes used by the kernels here are shared by both libraries, so this
// header selects the backend include and re-exports it under one name.
//
// Full functionality is required on both paths: device-, block- and
// warp-level APIs all come from the selected library, and the backend
// selection is made purely by the platform macro the toolchain defines.
// Code that needs a primitive absent from one library must not be added
// here behind a silent fallback; extend the wrapper explicitly instead.

#if defined(USE_ROCM)
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#else
#include <cub/cub.cuh>
#endif
