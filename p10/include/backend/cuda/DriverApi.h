#pragma once

// Driver entry points are resolved lazily through the runtime's loader so
// that CUDA wheels can be imported on machines without the NVIDIA driver.
// A null result means the symbol could not be resolved (driver missing or
// too old); callers must check before invoking.

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace tensorplay {
namespace cuda {
namespace driver {

template <typename Fn>
Fn resolve_symbol(const char* name) {
    void* symbol = nullptr;
    cudaDriverEntryPointQueryResult query{};
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12050)
    if (cudaGetDriverEntryPointByVersion(name, &symbol, 12000, cudaEnableDefault,
                                         &query) == cudaSuccess &&
        query == cudaDriverEntryPointSuccess && symbol != nullptr) {
        return reinterpret_cast<Fn>(symbol);
    }
#endif
#if defined(CUDA_VERSION) && (CUDA_VERSION < 13000)
    if (cudaGetDriverEntryPoint(name, &symbol, cudaEnableDefault, &query) ==
            cudaSuccess &&
        query == cudaDriverEntryPointSuccess && symbol != nullptr) {
        return reinterpret_cast<Fn>(symbol);
    }
#endif
    return nullptr;
}

}  // namespace driver
}  // namespace cuda
}  // namespace tensorplay
