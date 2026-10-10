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
#if defined(USE_ROCM)
    // The AMD runtime exposes a single unversioned query: it looks the
    // requested API base name up in the loaded runtime library, which is
    // the exact counterpart of the two CUDA queries below (there is no
    // versioned form and nothing is deprecated on this side).
    hipDriverEntryPointQueryResult query{};
    if (hipGetDriverEntryPoint(name, &symbol, hipEnableDefault, &query) ==
            hipSuccess &&
        query == hipDriverEntryPointSuccess && symbol != nullptr) {
        return reinterpret_cast<Fn>(symbol);
    }
#else
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
#endif
    return nullptr;
}

}  // namespace driver
}  // namespace cuda
}  // namespace tensorplay
