#pragma once

#include "Macros.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <cublas_v2.h>
#include <cublasLt.h>
#include <curand.h>
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
    static curandGenerator_t getCurandGenerator();
    static void manual_seed(uint64_t seed);
    static void manual_seed_all(uint64_t seed);
};

// True once this process has made a successful CUDA runtime call. Seeding and
// other bookkeeping use this to avoid initializing CUDA eagerly.
P10_API bool isCudaInitialized();

// True in a forked child that inherited initialized CUDA state; CUDA calls are
// unusable there and must be skipped (mirrors torch.cuda._is_in_bad_fork).
P10_API bool isInBadFork();

// Records that a CUDA runtime call succeeded. Called from checkCuda.
P10_API void noteCudaRuntimeCall();

} // namespace cuda
} // namespace tensorplay
