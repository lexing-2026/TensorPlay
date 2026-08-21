#include "CUDAContext.h"

#include "CUDARuntime.h"
#include "Device.h"
#include "Exception.h"

#include <atomic>
#include <string>
#include <unistd.h>

#include <unordered_map>

namespace tensorplay {
namespace cuda {
namespace {

// CUDA initialization tracking. A forked child inherits unusable CUDA state
// ("bad fork", torch terminology): runtime calls there fail with
// cudaErrorInitializationError. Seeding must therefore be lazy (never
// initialize CUDA) and skipped in bad-fork children, mirroring
// torch.cuda.manual_seed_all's _lazy_call + _is_in_bad_fork behavior.
std::atomic<bool> g_cuda_initialized{false};
std::atomic<pid_t> g_cuda_init_pid{0};
std::atomic<bool> g_has_pending_seed{false};
std::atomic<uint64_t> g_pending_seed{0};

void checkCublas(cublasStatus_t error, const char* operation) {
    if (error != CUBLAS_STATUS_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed with cuBLAS status " +
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

void checkCurand(curandStatus_t error, const char* operation) {
    if (error != CURAND_STATUS_SUCCESS) {
        TP_THROW(RuntimeError,
                 std::string(operation) + " failed with cuRAND status " +
                 std::to_string(static_cast<int>(error)));
    }
}

struct DeviceHandles {
    cublasHandle_t cublas = nullptr;
    cublasLtHandle_t cublas_lt = nullptr;
    curandGenerator_t curand = nullptr;
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
    checkCublas(cublasCreate(&created->cublas), "cublasCreate");
    checkCublas(cublasLtCreate(&created->cublas_lt), "cublasLtCreate");
    checkCurand(curandCreateGenerator(&created->curand, CURAND_RNG_PSEUDO_DEFAULT),
                "curandCreateGenerator");
    checkCurand(curandSetPseudoRandomGeneratorSeed(created->curand, 1234ULL),
                "curandSetPseudoRandomGeneratorSeed");
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

curandGenerator_t CUDAContext::getCurandGenerator() {
    auto& handles = handlesForCurrentDevice();
    checkCurand(curandSetStream(handles.curand, getCurrentCUDAStream().stream()),
                "curandSetStream");
    return handles.curand;
}

void CUDAContext::manual_seed(uint64_t seed) {
    auto stream = getCurrentCUDAStream();
    stream.synchronize();
    auto generator = getCurandGenerator();
    checkCurand(curandSetPseudoRandomGeneratorSeed(generator, seed),
                "curandSetPseudoRandomGeneratorSeed");
    checkCurand(curandSetGeneratorOffset(generator, 0), "curandSetGeneratorOffset");
}

void CUDAContext::manual_seed_all(uint64_t seed) {
    const int count = deviceCount();
    for (int device = 0; device < count; ++device) {
        CUDAGuard guard(device);
        manual_seed(seed);
    }
}

void manual_seed(uint64_t seed) {
    // Lazy: never initialize CUDA from a seeding call. If CUDA is not
    // initialized yet, stash the seed and apply it at first real CUDA use.
    if (!isCudaInitialized()) {
        g_pending_seed.store(seed, std::memory_order_relaxed);
        g_has_pending_seed.store(true, std::memory_order_release);
        return;
    }
    if (isInBadFork()) return;
    CUDAContext::manual_seed(seed);
}

void manual_seed_all(uint64_t seed) {
    if (!isCudaInitialized()) {
        g_pending_seed.store(seed, std::memory_order_relaxed);
        g_has_pending_seed.store(true, std::memory_order_release);
        return;
    }
    if (isInBadFork()) return;
    CUDAContext::manual_seed_all(seed);
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
    if (g_has_pending_seed.exchange(false, std::memory_order_acq_rel)) {
        CUDAContext::manual_seed_all(g_pending_seed.load(std::memory_order_relaxed));
    }
}

} // namespace cuda
} // namespace tensorplay
