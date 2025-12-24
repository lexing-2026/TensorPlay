#include "CUDAContext.h"
#include "Exception.h"
#include "Device.h"
#include <mutex>

namespace tensorplay {
namespace cuda {

namespace {
    void checkCudaError(cudaError_t err, const char* msg) {
        if (err != cudaSuccess) {
            TP_THROW(RuntimeError, std::string(msg) + ": " + cudaGetErrorString(err));
        }
    }
    
    void checkCublasError(cublasStatus_t err, const char* msg) {
        if (err != CUBLAS_STATUS_SUCCESS) {
            TP_THROW(RuntimeError, std::string(msg) + " failed");
        }
    }

#ifdef USE_CUDNN
    void checkCudnnError(cudnnStatus_t err, const char* msg) {
        if (err != CUDNN_STATUS_SUCCESS) {
            TP_THROW(RuntimeError, std::string(msg) + ": " + cudnnGetErrorString(err));
        }
    }
#endif

    void checkCurandError(curandStatus_t err, const char* msg) {
        if (err != CURAND_STATUS_SUCCESS) {
            TP_THROW(RuntimeError, std::string(msg) + " failed");
        }
    }
}

#ifdef USE_CUDNN
cudnnHandle_t CUDAContext::getCudnnHandle() {
    static cudnnHandle_t handle;
    static std::once_flag flag;
    std::call_once(flag, []() {
        checkCudnnError(cudnnCreate(&handle), "cudnnCreate");
    });
    return handle;
}
#endif

cublasHandle_t CUDAContext::getCublasHandle() {
    static cublasHandle_t handle;
    static std::once_flag flag;
    std::call_once(flag, []() {
        checkCublasError(cublasCreate(&handle), "cublasCreate");
    });
    return handle;
}

curandGenerator_t CUDAContext::getCurandGenerator() {
    static curandGenerator_t generator;
    static std::once_flag flag;
    std::call_once(flag, []() {
        checkCurandError(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT), "curandCreateGenerator");
        checkCurandError(curandSetPseudoRandomGeneratorSeed(generator, 1234ULL), "curandSetPseudoRandomGeneratorSeed");
    });
    return generator;
}

void CUDAContext::manual_seed(uint64_t seed) {
    auto gen = getCurandGenerator();
    checkCurandError(curandSetPseudoRandomGeneratorSeed(gen, seed), "curandSetPseudoRandomGeneratorSeed");
}

void manual_seed(uint64_t seed) {
    CUDAContext::manual_seed(seed);
}

} // namespace cuda
} // namespace tensorplay
