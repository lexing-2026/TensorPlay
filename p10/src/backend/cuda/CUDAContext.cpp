#include "CUDAContext.h"

#include "CUDAGenerator.h"
#include "CUDARuntime.h"
#include "Device.h"
#include "Exception.h"

#include <atomic>
#include <string>

#ifdef _WIN32
#include <process.h>  // getpid via _getpid
#define getpid _getpid
#else
#include <unistd.h>
#endif

#include <unordered_map>

namespace tensorplay {
namespace cuda {
namespace {

// CUDA initialization tracking. A forked child inherits unusable CUDA state
// cudaErrorInitializationError. Seeding must therefore be lazy (never
// initialize CUDA) and skipped in bad-fork children, preserving
std::atomic<bool> g_cuda_initialized{false};
std::atomic<pid_t> g_cuda_init_pid{0};

void checkCublas(cublasStatus_t error, const char* operation) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed with cuBLAS status " +
                 std::to_string(static_cast<int>(error)));
    }
}

void checkCusolver(cusolverStatus_t error, const char* operation) {
    if (error != CUSOLVER_STATUS_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed: cusolver error " +
                     std::to_string(static_cast<int>(error)));
    }
}

#ifdef USE_CUDNN
void checkCudnn(cudnnStatus_t error, const char* operation) {
    if (error != CUDNN_STATUS_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed: " + cudnnGetErrorString(error));
    }
}
#endif

struct DeviceHandles {
    cusolverDnHandle_t cusolver_dn = nullptr;
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
#ifdef USE_CUDNN
    cudnnHandle_t cudnn = nullptr;
#endif
};

DeviceHandles& handlesForCurrentDevice() {
    // Handles carry mutable stream state. Keeping one set per OS thread and
    // device avoids races when independent Python/C++ threads select different
    // current streams. They intentionally live until process exit.
    static thread_local std::unordered_map<int, DeviceHandles*> handles;
    const int device = currentDevice();
    auto it = handles.find(device);
    if (it != handles.end()) return *it->second;

    auto* created = new DeviceHandles();
    checkCusolver(cusolverDnCreate(&created->cusolver_dn), "cusolverDnCreate");
    checkCublas(cublasCreate(&created->cublas), "cublasCreate");
    checkCublas(cublasLtCreate(&created->cublas_lt), "cublasLtCreate");
#ifdef USE_CUDNN
    checkCudnn(cudnnCreate(&created->cudnn), "cudnnCreate");
#endif
    handles.emplace(device, created);
    return *created;
}

} // namespace

#ifdef USE_CUDNN
cudnnHandle_t CUDAContext::getCudnnHandle() {
    auto& handles = handlesForCurrentDevice();
    checkCudnn(cudnnSetStream(handles.cudnn, getCurrentCUDAStream().stream()),
               "cudnnSetStream");
    return handles.cudnn;
}
#endif

cublasHandle_t CUDAContext::getCublasHandle() {
    auto& handles = handlesForCurrentDevice();
    checkCublas(cublasSetStream(handles.cublas, getCurrentCUDAStream().stream()),
                "cublasSetStream");
    return handles.cublas;
}

cublasLtHandle_t CUDAContext::getCublasLtHandle() {
    return handlesForCurrentDevice().cublas_lt;
}

cusolverDnHandle_t CUDAContext::getCusolverDnHandle() {
    auto& handles = handlesForCurrentDevice();
    checkCusolver(cusolverDnSetStream(handles.cusolver_dn, getCurrentCUDAStream().stream()),
                  "cusolverDnSetStream");
    return handles.cusolver_dn;
}

void CUDAContext::warmupHandles() {
    handlesForCurrentDevice();
}

bool isCudaInitialized() {
    return g_cuda_initialized.load(std::memory_order_acquire);
}

bool isInBadFork() {
    return isCudaInitialized() &&
           getpid() != g_cuda_init_pid.load(std::memory_order_relaxed);
}

void noteCudaRuntimeCall() {
    bool expected = false;
    if (!g_cuda_initialized.compare_exchange_strong(expected, true,
                                                    std::memory_order_acq_rel)) {
        return;
    }
    g_cuda_init_pid.store(getpid(), std::memory_order_relaxed);
    // Apply a seed stashed by a pre-initialization manual_seed call.
    apply_pending_seed();
}

} // namespace cuda
} // namespace tensorplay
