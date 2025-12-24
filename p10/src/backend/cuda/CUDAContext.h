#pragma once

#include "Macros.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#endif
#include <cublas_v2.h>
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
    static curandGenerator_t getCurandGenerator();
    static void manual_seed(uint64_t seed);
};

} // namespace cuda
} // namespace tensorplay
