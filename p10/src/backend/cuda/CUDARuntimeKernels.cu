#include "CUDARuntime.h"

#ifdef USE_CUDA

namespace tensorplay {
namespace cuda {
namespace {

__global__ void sleepKernel(uint64_t cycles) {
    const uint64_t start = clock64();
    while (clock64() - start < cycles) {
    }
}

} // namespace

void sleep(uint64_t cycles) {
    sleepKernel<<<1, 1, 0, getCurrentCUDAStream().stream()>>>(cycles);
    checkCuda(cudaGetLastError(), "CUDA sleep kernel launch");
}

} // namespace cuda
} // namespace tensorplay

#endif
