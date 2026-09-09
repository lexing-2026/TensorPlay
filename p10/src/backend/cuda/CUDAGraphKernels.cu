#include "Exception.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace tensorplay {
namespace cuda {
namespace graph {

namespace {

#if !defined(USE_ROCM) && CUDART_VERSION >= 12040
// Captured into the parent graph right before a conditional node: at replay
// time this device-side write decides whether the node's body executes.
__global__ void set_conditional_handle_kernel(
    cudaGraphConditionalHandle handle, const bool* value) {
    cudaGraphSetConditional(handle, *value);
}
#endif

} // namespace

bool conditionalNodesSupported() {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12040
    return true;
#else
    return false;
#endif
}

void launchSetConditionalHandle(uint64_t handle, const void* pred_bool,
                                cudaStream_t stream) {
#if !defined(USE_ROCM) && CUDART_VERSION >= 12040
    set_conditional_handle_kernel<<<1, 1, 0, stream>>>(
        static_cast<cudaGraphConditionalHandle>(handle),
        static_cast<const bool*>(pred_bool));
    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        TP_THROW(RuntimeError,
                 std::string("set_conditional_handle_kernel launch failed: ") +
                     cudaGetErrorString(error));
    }
#else
    (void)handle;
    (void)pred_bool;
    (void)stream;
    TP_THROW(RuntimeError,
             "CUDA graphs conditional nodes require CUDA >= 12.4");
#endif
}

} // namespace graph
} // namespace cuda
} // namespace tensorplay
