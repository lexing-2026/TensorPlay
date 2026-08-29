#pragma once

#include "Macros.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

namespace tensorplay {
namespace cuda {

class P10_API CUDAContext {
public:
#ifdef USE_CUDNN
    static cudnnHandle_t getCudnnHandle();
#endif
    static cublasHandle_t getCublasHandle();
    static cublasLtHandle_t getCublasLtHandle();
    static cusolverDnHandle_t getCusolverDnHandle();
    // Creates every lazy library handle for the current device up front.
    // Handle creation performs internal allocations, which are illegal once
    // CUDAGraph::capture_begin calls this before cudaStreamBeginCapture.
    static void warmupHandles();
};

// True once this process has made a successful CUDA runtime call. Seeding and
// other bookkeeping use this to avoid initializing CUDA eagerly.
P10_API bool isCudaInitialized();

// True in a forked child that inherited initialized CUDA state; CUDA calls are
P10_API bool isInBadFork();

// Records that a CUDA runtime call succeeded. Called from checkCuda.
P10_API void noteCudaRuntimeCall();

} // namespace cuda
} // namespace tensorplay
