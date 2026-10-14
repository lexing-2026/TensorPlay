// Device kernels for the integer matrix product.
//
// _int_mm multiplies two 8-bit matrices and accumulates in 32 bits, the shape
// a quantized linear layer needs before its requantization step.  The signed
// int8 x int8 product is served by the cuBLASLt int8 tensor-core GEMM with an
// int32 accumulator; the tile kernel below is the fallback for the unsigned
// left operand and for shapes the tensor-core path does not accept (rows at
// or below 16, or a reduction/output axis that is not a multiple of 8).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "CUDAContext.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

#define TP_INT_MM_CHECK(expr)                                                  \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            TP_THROW(RuntimeError, std::string("CUDA Error: ") +               \
                                       cudaGetErrorString(status));            \
        }                                                                      \
    } while (0)

#define CUBLASLT_CHECK(condition)                                              \
    do {                                                                       \
        const cublasStatus_t _tp_cublaslt_status = (condition);                \
        if (_tp_cublaslt_status != CUBLAS_STATUS_SUCCESS) {                    \
            TP_THROW(RuntimeError, "cuBLASLt Error " +                         \
                                     std::to_string(                           \
                                         static_cast<int>(_tp_cublaslt_status))); \
        }                                                                      \
    } while (0)

constexpr int kIntMMTile = 16;

// Fallback tile kernel: stages a square block of each operand in shared memory
// so every 8-bit value loaded from global memory is reused across the tile,
// and the accumulator stays in a register for the whole K sweep.  The tiles
// keep the 8-bit payload (with one padding column per row to spread bank
// accesses), so four times as many elements fit in the same shared-memory
// budget as a widened staging.
template <typename lhs_t>
__global__ void int_mm_kernel(int64_t M, int64_t K, int64_t N,
                              const lhs_t* __restrict__ a,
                              const int8_t* __restrict__ b,
                              int32_t* __restrict__ c) {
    __shared__ lhs_t a_tile[kIntMMTile][kIntMMTile + 1];
    __shared__ int8_t b_tile[kIntMMTile][kIntMMTile + 1];

    const int64_t row = static_cast<int64_t>(blockIdx.y) * kIntMMTile + threadIdx.y;
    const int64_t col = static_cast<int64_t>(blockIdx.x) * kIntMMTile + threadIdx.x;
    int32_t acc = 0;

    const int64_t tiles = (K + kIntMMTile - 1) / kIntMMTile;
    for (int64_t t = 0; t < tiles; ++t) {
        const int64_t a_col = t * kIntMMTile + threadIdx.x;
        const int64_t b_row = t * kIntMMTile + threadIdx.y;
        a_tile[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? a[row * K + a_col] : 0;
        b_tile[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? b[b_row * N + col] : 0;
        __syncthreads();

        for (int i = 0; i < kIntMMTile; ++i) {
            acc += static_cast<int32_t>(a_tile[threadIdx.y][i]) *
                   static_cast<int32_t>(b_tile[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// cuBLASLt int8 tensor-core path.
//
// The row-major product C_r(M,N) = A_r(M,K) * B_r(K,N) is expressed in
// cuBLASLt's native column-major space with the transpose trick:
//   C_r^T = B_r^T * A_r^T  =>  col-major C_c (N,M) = B_c (N,K) * A_c (K,M),
// with B_c = B_r^T (ld = N), A_c = A_r^T (ld = K), C_c = C_r^T (ld = N).
// The int32 accumulation is selected with compute type CUBLAS_COMPUTE_32I and
// scale type CUDA_R_32I; alpha and beta are int32 scalars.  The same plan
// descriptors are reused for every call with a given shape on a device.
// ---------------------------------------------------------------------------

struct Int8GemmKey {
    int device;
    int64_t m, n, k;
    bool operator==(const Int8GemmKey& o) const {
        return device == o.device && m == o.m && n == o.n && k == o.k;
    }
};

struct Int8GemmKeyHash {
    size_t operator()(const Int8GemmKey& k) const {
        size_t h = std::hash<int64_t>()(k.m);
        h ^= std::hash<int64_t>()(k.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(k.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct Int8GemmPlan {
    int device = -1;
    std::mutex execution_mutex;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    // Candidate algorithms returned by the heuristic query.  The first call
    // times them back-to-back and caches the measured winner in `algo`, so
    // every later matmul on this shape reuses the same algorithm.
    static constexpr int kMaxCandidates = 8;
    cublasLtMatmulHeuristicResult_t candidates[kMaxCandidates];
    int candidate_count = 0;
    bool autotuned = false;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace_size = 0;
    size_t workspace_limit = 32 * 1024 * 1024;

    ~Int8GemmPlan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    }
};

std::mutex& int8_plan_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::unordered_map<Int8GemmKey, std::shared_ptr<Int8GemmPlan>, Int8GemmKeyHash>&
int8_plan_cache() {
    static auto* cache = new std::unordered_map<
        Int8GemmKey, std::shared_ptr<Int8GemmPlan>, Int8GemmKeyHash>();
    return *cache;
}

std::shared_ptr<Int8GemmPlan> get_int8_gemm_plan(int64_t M, int64_t N,
                                                int64_t K) {
    const int device = currentDevice();
    const Int8GemmKey key{device, N, M, K};
    std::lock_guard<std::mutex> lock(int8_plan_mutex());
    auto& cache = int8_plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto plan = std::make_shared<Int8GemmPlan>();
    plan->device = device;

    constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32I;
    constexpr cudaDataType_t scale_type = CUDA_R_32I;
    constexpr cudaDataType_t ab_type = CUDA_R_8I;
    constexpr cudaDataType_t c_type = CUDA_R_32I;

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type,
                                            scale_type));
    cublasOperation_t op_n = CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_n, sizeof(op_n)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));

    // Natural row-major layouts: C(M,N) = A(M,K) @ B(K,N), with every matrix
    // described in row-major order so the library sees the same access
    // pattern the caller has and can pick its fastest int8 algorithm.
    constexpr cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, ab_type, M, K, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan->a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, ab_type, K, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan->b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, c_type, M, N, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(
        plan->c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order)));

    // Reserve a scratch budget and ask the library for the best tensor-core
    // algorithms for this exact shape.  Without a heuristic pick the library
    // falls back to a generic algorithm that leaves a large fraction of the
    // int8 tensor-core throughput unused.
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&plan->pref));
    size_t workspace_size = plan->workspace_limit;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
        plan->pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_size, sizeof(workspace_size)));

    constexpr int kMaxCandidates = Int8GemmPlan::kMaxCandidates;
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, plan->a_desc,
        plan->b_desc, plan->c_desc, plan->c_desc, plan->pref, kMaxCandidates,
        plan->candidates, &returned));
    if (returned == 0) {
        TP_THROW(RuntimeError, "cuBLASLt: no int8 heuristic algorithm found");
    }
    plan->candidate_count = returned;
    for (int i = 0; i < plan->candidate_count; ++i) {
        plan->workspace_size =
            std::max(plan->workspace_size, plan->candidates[i].workspaceSize);
    }

    cache.emplace(key, plan);
    return plan;
}

// Runs the signed product through cuBLASLt.  Returns false when the library
// rejects the shape or the operand alignment, in which case the caller falls
// back to the tile kernel.
// One shared scratch buffer per (device, stream).  cuBLASLt treats the
// workspace as scratch for a single enqueued matmul and operations on one
// stream serialize, so every int8 plan on a stream can share that stream's
// buffer.
void* shared_int8_workspace(int device, size_t bytes) {
    struct Entry {
        void* ptr = nullptr;
        size_t size = 0;
    };
    static std::mutex ws_mutex;
    static auto* table =
        new std::unordered_map<uint64_t, Entry>();
    void* stream = getCurrentCUDAStream().stream();
    const uint64_t key = (static_cast<uint64_t>(device) << 32) ^
                         reinterpret_cast<uintptr_t>(stream);
    std::lock_guard<std::mutex> lock(ws_mutex);
    Entry& entry = (*table)[key];
    if (entry.size < bytes) {
        if (entry.ptr) {
            checkCuda(cudaFree(entry.ptr), "int8 workspace free");
        }
        checkCuda(cudaMalloc(&entry.ptr, bytes), "int8 workspace alloc");
        entry.size = bytes;
    }
    return entry.ptr;
}

// Times `samples` back-to-back executions of one candidate after a single
// untimed warm-up that doubles as the support probe.  Returns the average
// milliseconds per execution, or a negative value when the candidate failed
// to run at all.
template <typename Exec>
double time_int8_candidate(const Exec& exec, const cublasLtMatmulAlgo_t& algo,
                           int samples, cudaStream_t stream) {
    cudaEvent_t ev_start = nullptr, ev_end = nullptr;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_end);
    if (exec(algo) != CUBLAS_STATUS_SUCCESS) {
        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_end);
        return -1.0;
    }
    cudaEventRecord(ev_start, stream);
    for (int t = 0; t < samples; ++t) {
        exec(algo);
    }
    cudaEventRecord(ev_end, stream);
    cudaEventSynchronize(ev_end);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, ev_start, ev_end);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    return static_cast<double>(ms) / samples;
}

bool int8_gemm_cublaslt(const Tensor& self, const Tensor& mat2,
                        Tensor& result) {
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    auto plan = get_int8_gemm_plan(M, N, K);
    std::lock_guard<std::mutex> execution_lock(plan->execution_mutex);
    int32_t alpha = 1;
    int32_t beta = 0;
    void* workspace = shared_int8_workspace(plan->device, plan->workspace_size);
    cudaStream_t stream = getCurrentCUDAStream().stream();

    if (!plan->autotuned) {
        // Time each heuristic candidate with beta = 0 and keep the measured
        // winner for all later calls on this shape.  The timing runs write
        // into `result`, which the final matmul (also beta = 0) overwrites.
        constexpr int kSamples = 10;
        int best = -1;
        double best_ms = 1e300;
        for (int i = 0; i < plan->candidate_count; ++i) {
            const double ms = time_int8_candidate(
                [&](const cublasLtMatmulAlgo_t& algo) {
                    return cublasLtMatmul(
                        CUDAContext::getCublasLtHandle(), plan->matmul_desc,
                        &alpha, self.const_data_ptr<int8_t>(), plan->a_desc,
                        mat2.const_data_ptr<int8_t>(), plan->b_desc, &beta,
                        result.data_ptr<int32_t>(), plan->c_desc,
                        result.data_ptr<int32_t>(), plan->c_desc, &algo,
                        workspace, plan->workspace_size, stream);
                },
                plan->candidates[i].algo, kSamples, stream);
            if (ms >= 0 && ms < best_ms) {
                best_ms = ms;
                best = i;
            }
        }
        if (best < 0) {
            return false;
        }
        plan->algo = plan->candidates[best].algo;
        plan->autotuned = true;
    }

    const cublasStatus_t status = cublasLtMatmul(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, &alpha,
        self.const_data_ptr<int8_t>(), plan->a_desc,
        mat2.const_data_ptr<int8_t>(), plan->b_desc, &beta,
        result.data_ptr<int32_t>(), plan->c_desc,
        result.data_ptr<int32_t>(), plan->c_desc, &plan->algo, workspace,
        plan->workspace_size, stream);
    return status == CUBLAS_STATUS_SUCCESS;
}

bool use_int8_tensor_cores(int64_t M, int64_t K, int64_t N) {
    return M > 16 && K % 8 == 0 && N % 8 == 0;
}

void check_int_mm_operands(const Tensor& self, const Tensor& mat2) {
    TP_CHECK(self.dim() == 2 && mat2.dim() == 2,
             "_int_mm: expected two matrices but got ", self.dim(), "-D and ",
             mat2.dim(), "-D tensors");
    TP_CHECK(self.size(1) == mat2.size(0),
             "_int_mm: cannot multiply a ", self.size(0), " by ", self.size(1),
             " matrix with a ", mat2.size(0), " by ", mat2.size(1), " matrix");
    TP_CHECK(self.dtype() == DType::Int8 || self.dtype() == DType::UInt8,
             "_int_mm: expected the left operand to be int8 or uint8");
    TP_CHECK(mat2.dtype() == DType::Int8,
             "_int_mm: expected the right operand to be int8");
}

// Fast path for a live transposed weight view.  `mat2` is the non-contiguous
// [K,N] view of a contiguous [N,K] int8 weight (strides [1,K]); the GEMM
// `self(M,K) @ mat2(K,N)` is fed to cuBLAS as `weight^T @ activation` in the
// column-major space the int8 tensor-core kernels are tuned for, without
// materializing a [K,N] copy.
bool int8_gemm_transposed_ex(const Tensor& self, const Tensor& mat2,
                             Tensor& result) {
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    int32_t alpha = 1;
    int32_t beta = 0;
    const cublasStatus_t status = cublasGemmEx(
        CUDAContext::getCublasHandle(), CUBLAS_OP_T, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        &alpha, mat2.data_ptr(), CUDA_R_8I, static_cast<int>(K),
        self.data_ptr(), CUDA_R_8I, static_cast<int>(K), &beta,
        result.data_ptr(), CUDA_R_32I, static_cast<int>(N),
        CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS;
}

void int_mm_impl(const Tensor& self, const Tensor& mat2, Tensor& out) {
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    if (out.numel() == 0) return;
    if (K == 0) {
        checkCuda(cudaMemsetAsync(out.data_ptr(), 0,
                                  out.numel() * sizeof(int32_t),
                                  getCurrentCUDAStream().stream()),
                  "int_mm zero fill");
        return;
    }

    const Tensor lhs = self.contiguous();

    // The fast tensor-core path accepts a live transposed weight view
    // directly, avoiding the [K,N] copy the generic path would make.
    const bool mat2_is_transposed_view =
        !mat2.is_contiguous() && mat2.dim() == 2 &&
        mat2.stride(0) == 1 && mat2.stride(1) == K;
    if (self.dtype() == DType::Int8 && mat2_is_transposed_view &&
        use_int8_tensor_cores(M, K, N) &&
        int8_gemm_transposed_ex(lhs, mat2, out)) {
        return;
    }

    const Tensor rhs = mat2.contiguous();
    if (self.dtype() == DType::Int8 && use_int8_tensor_cores(M, K, N) &&
        int8_gemm_cublaslt(lhs, rhs, out)) {
        return;
    }

    const dim3 block(kIntMMTile, kIntMMTile);
    const dim3 grid(static_cast<unsigned>((N + kIntMMTile - 1) / kIntMMTile),
                    static_cast<unsigned>((M + kIntMMTile - 1) / kIntMMTile));
    auto stream = getCurrentCUDAStream().stream();
    if (self.dtype() == DType::Int8) {
        int_mm_kernel<int8_t><<<grid, block, 0, stream>>>(
            M, K, N, lhs.data_ptr<int8_t>(), rhs.data_ptr<int8_t>(),
            out.data_ptr<int32_t>());
    } else {
        int_mm_kernel<uint8_t><<<grid, block, 0, stream>>>(
            M, K, N, lhs.data_ptr<uint8_t>(), rhs.data_ptr<int8_t>(),
            out.data_ptr<int32_t>());
    }
    TP_INT_MM_CHECK(cudaGetLastError());
}

}  // namespace

Tensor _int_mm_cuda(const Tensor& self, const Tensor& mat2) {
    check_int_mm_operands(self, mat2);
    const int64_t M = self.size(0);
    const int64_t K = self.size(1);
    const int64_t N = mat2.size(1);
    Tensor out = Tensor::empty({M, N}, DType::Int32, self.device());
    int_mm_impl(self, mat2, out);
    return out;
}

Tensor& _int_mm_out_cuda(const Tensor& self, const Tensor& mat2, Tensor& out) {
    check_int_mm_operands(self, mat2);
    TP_CHECK(out.dtype() == DType::Int32,
             "_int_mm: expected an int32 destination");
    const std::vector<int64_t> target{self.size(0), mat2.size(1)};
    if (static_cast<std::vector<int64_t>>(out.shape()) != target) {
        out.resize_(target);
    }
    if (out.numel() == 0) return out;
    if (out.is_contiguous()) {
        int_mm_impl(self, mat2, out);
        return out;
    }
    Tensor value =
        Tensor::empty({self.size(0), mat2.size(1)}, DType::Int32, self.device());
    int_mm_impl(self, mat2, value);
    out.copy_(value);
    return out;
}

#undef TP_INT_MM_CHECK
#undef CUBLASLT_CHECK

TENSORPLAY_LIBRARY_IMPL(CUDA, IntMMKernels) {
    m.impl("_int_mm", _int_mm_cuda);
    m.impl("_int_mm.out", _int_mm_out_cuda);
}

}  // namespace cuda
}  // namespace tensorplay