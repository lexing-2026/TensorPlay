// Host-side cuBLAS/cuBLASLt GEMM orchestration.
//
// This lives in a plain C++ translation unit on purpose: none of this code
// compiles device kernels, and keeping it out of nvcc avoids toolchain
// fragility (nvcc internal errors while instantiating host-only templates).
// Kernels in LinearAlgebraKernels.cu call into these entry points.

#include "CudaGemm.h"
#include "CUDAContext.h"
#include "CUDAGraph.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Context.h"
#include "LinearAlgebraNames.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUBLAS_CHECK(condition) \
  do { \
    const cublasStatus_t _tp_cublas_status = (condition); \
    if (_tp_cublas_status != CUBLAS_STATUS_SUCCESS) { \
      TP_THROW(RuntimeError, "cuBLAS Error " + std::to_string(static_cast<int>(_tp_cublas_status))); \
    } \
  } while (0)

#define CUBLASLT_CHECK(condition) \
  do { \
    const cublasStatus_t _tp_cublaslt_status = (condition); \
    if (_tp_cublaslt_status != CUBLAS_STATUS_SUCCESS) { \
      TP_THROW(RuntimeError, "cuBLASLt Error " + std::to_string(static_cast<int>(_tp_cublaslt_status))); \
    } \
  } while (0)

namespace {

Tensor& zero_matmul_output(Tensor& output) {
    if (output.numel() != 0) {
        checkCuda(cudaMemsetAsync(
            output.data_ptr(),
            0,
            output.numel() * output.itemsize(),
            getCurrentCUDAStream().stream()),
            "matmul zero output");
    }
    return output;
}

// Row-major GEMM via cuBLASLt (column-major native).
// C_r(M x N) = A_r(M x K) * B_r(K x N). Using the transpose trick:
//   C_r^T = B_r^T * A_r^T  =>  col-major C_c (N x M) = B_c (N x K) * A_c (K x M)
// with B_c = B_r^T (ld = N), A_c = A_r^T (ld = K), C_c = C_r^T (ld = N).
struct GemmKey {
    ScalarType dtype;
    int64_t m, n, k;
    int device;
    bool has_bias;
    bool other_transposed;

    bool operator==(const GemmKey& o) const {
        return dtype == o.dtype && m == o.m && n == o.n && k == o.k &&
               device == o.device && has_bias == o.has_bias &&
               other_transposed == o.other_transposed;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey& k) const {
        size_t h = std::hash<int64_t>()(k.m);
        h ^= std::hash<int64_t>()(k.n) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(k.k) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(static_cast<int>(k.dtype)) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.device) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(k.has_bias) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>()(k.other_transposed) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct GemmPlan {
    int device = -1;
    std::mutex execution_mutex;
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr;
    cublasLtMatrixLayout_t b_desc = nullptr;
    cublasLtMatrixLayout_t c_desc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    // Candidate algorithms from the heuristic query.  The first call runs a
    // one-time micro-autotune over these and reorders them so index 0 is the
    // measured winner; later calls reuse it.
    std::vector<cublasLtMatmulHeuristicResult_t> candidates;
    bool autotuned = false;
    size_t workspace_size = 0;
    // Owned by workspace_registry(); entries are never erased so pointers
    // stay valid for the process lifetime.
    void* workspace_ptr = nullptr;

    ~GemmPlan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    }
};

std::vector<Tensor>& workspace_registry() {
    static auto* registry = new std::vector<Tensor>();
    return *registry;
}

std::mutex& plan_mutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::unordered_map<GemmKey, std::shared_ptr<GemmPlan>, GemmKeyHash>& plan_cache() {
    static auto* cache = new std::unordered_map<GemmKey, std::shared_ptr<GemmPlan>, GemmKeyHash>();
    return *cache;
}

cudaDataType_t to_cublas_type(DType t) {
    switch (t) {
        case DType::Float32: return CUDA_R_32F;
        case DType::Float64: return CUDA_R_64F;
        case DType::Float16: return CUDA_R_16F;
        case DType::BFloat16: return CUDA_R_16BF;
        case DType::ComplexFloat: return CUDA_C_32F;
        case DType::ComplexDouble: return CUDA_C_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

cublasComputeType_t to_compute_type(DType t) {
    switch (t) {
        // PyTorch's default is torch.backends.cuda.matmul.allow_tf32=False
        // (float32_matmul_precision == "highest"), and its CUDABlas helpers
        // accumulate Half/BFloat16 in FP32 with float alpha/beta.  Mirror
        // that contract; "high"/"medium" enable TF32 compute for Float32,
        // matching Context::allowTF32CuBLAS.
        case DType::Float32:
            if (globalContext().allowTF32CuBLAS()) return CUBLAS_COMPUTE_32F_FAST_TF32;
            return CUBLAS_COMPUTE_32F;
        case DType::Float64: return CUBLAS_COMPUTE_64F;
        case DType::Float16: return CUBLAS_COMPUTE_32F;
        case DType::BFloat16: return CUBLAS_COMPUTE_32F;
        case DType::ComplexFloat: return CUBLAS_COMPUTE_32F;
        case DType::ComplexDouble: return CUBLAS_COMPUTE_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

cudaDataType_t to_scale_type(DType t) {
    switch (t) {
        case DType::Float32: return CUDA_R_32F;
        case DType::Float64: return CUDA_R_64F;
        // FP32 compute for reduced-precision inputs: scale in FP32.
        case DType::Float16: return CUDA_R_32F;
        case DType::BFloat16: return CUDA_R_32F;
        case DType::ComplexFloat: return CUDA_C_32F;
        case DType::ComplexDouble: return CUDA_C_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

// Alpha/beta must live in the scale type of the compute type.  Two slots:
// 0 = alpha, 1 = beta (autotune temporarily reuses slot 1 for zero).
void* to_scalar_ptr(double v, DType t, int slot) {
    static thread_local float f32[2];
    static thread_local double f64[2];
    static thread_local cuFloatComplex c32[2];
    static thread_local cuDoubleComplex c64[2];
    switch (t) {
        case DType::Float32: f32[slot] = static_cast<float>(v); return &f32[slot];
        case DType::Float64: f64[slot] = v; return &f64[slot];
        case DType::Float16: f32[slot] = static_cast<float>(v); return &f32[slot];
        case DType::BFloat16: f32[slot] = static_cast<float>(v); return &f32[slot];
        case DType::ComplexFloat: c32[slot] = make_cuFloatComplex(static_cast<float>(v), 0.0f); return &c32[slot];
        case DType::ComplexDouble: c64[slot] = make_cuDoubleComplex(v, 0.0); return &c64[slot];
        default: return nullptr;
    }
}

std::shared_ptr<GemmPlan> get_gemm_plan(DType dtype, int64_t M, int64_t N, int64_t K,
                                        bool has_bias, bool other_transposed) {
    const int device = currentDevice();
    GemmKey key{dtype, M, N, K, device, has_bias, other_transposed};
    std::lock_guard<std::mutex> lock(plan_mutex());
    auto& cache = plan_cache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto plan = std::make_shared<GemmPlan>();
    plan->device = device;
    cudaDataType_t cuda_type = to_cublas_type(dtype);
    cublasComputeType_t compute_type = to_compute_type(dtype);
    cudaDataType_t scale_type = to_scale_type(dtype);

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan->matmul_desc, compute_type, scale_type));
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;

    // Row-major GEMM expressed in cuBLAS's native column-major space.
    //
    // Plain path, C_r(M,N) = A_r(M,K) * B_r(K,N):
    //   B'  = B_r read as col-major (N,K), ld=N, op=N
    //   A'' = A_r read as col-major (K,M), ld=K, op=N
    //
    // Transposed-weights path (the x @ W.t() linear-layer pattern), where
    // `other` is a contiguous [N,K] row-major weight viewed as its transpose:
    //   W stored row-major (N,K) == col-major (K,N), ld=K, op=T -> (N,K)
    cublasOperation_t trans_a = other_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_n, sizeof(op_n)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    if (other_transposed) {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, cuda_type, K, N, K));
    } else {
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, cuda_type, N, K, N));
    }
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, cuda_type, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, cuda_type, N, M, N));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&plan->pref));
    size_t workspace_size = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(plan->pref,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspace_size, sizeof(workspace_size)));
    plan->workspace_size = workspace_size;

    // Ask for several candidates; the first execution micro-autotunes them.
    constexpr int kMaxCandidates = 8;
    plan->candidates.resize(kMaxCandidates);
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, plan->a_desc, plan->b_desc, plan->c_desc, plan->c_desc,
        plan->pref, kMaxCandidates, plan->candidates.data(), &returned));
    if (returned == 0) {
        TP_THROW(RuntimeError, "cuBLASLt: no heuristic algorithm found");
    }
    plan->candidates.resize(returned);

    workspace_registry().push_back(
        Tensor({static_cast<int64_t>(workspace_size)}, DType::UInt8,
               Device(DeviceType::CUDA, device)));
    plan->workspace_ptr = workspace_registry().back().data_ptr();

    cache.emplace(key, plan);
    return plan;
}

} // namespace

void check_cublas_gemm_dtype(DType t) {
    switch (t) {
        case DType::Float32:
        case DType::Float64:
        case DType::Float16:
        case DType::BFloat16:
        case DType::ComplexFloat:
        case DType::ComplexDouble:
            return;
        default:
            // Torch wording: CUDA matmul rejects non-floating dtypes through
            // addmm_cuda ("addmm_cuda" not implemented for 'Int' etc.), even
            // when the mathematical result would be empty.
            TP_THROW(NotImplementedError, "\"addmm_cuda\" not implemented for '",
                     pretty_dtype_name(t), "'");
    }
}

Tensor& zero_matmul_output_cuda(Tensor& output) {
    return zero_matmul_output(output);
}

void gemm_impl(const Tensor& self, const Tensor& other, Tensor& result,
               double alpha, double beta, const Tensor* bias) {
    if (self.dim() != 2 || other.dim() != 2) {
        TP_THROW(RuntimeError, "mm: tensors must be 2D");
    }
    if (self.shape()[1] != other.shape()[0]) {
        TP_THROW(RuntimeError, "mm: shape mismatch");
    }

    const auto self_strides = self.strides();
    const bool self_transposed_contiguous =
        !self.is_contiguous() && self.dim() == 2 &&
        self_strides[0] == 1 && self_strides[1] == self.shape()[0];
    const bool native_cublas_dtype =
        isComplexType(self.dtype()) || self.dtype() == DType::Float16 ||
        self.dtype() == DType::BFloat16;
    // A live transpose view is already a valid column-major operand for the
    // row-major cuBLAS trick.  Keep it in place for the native cuBLAS dtypes;
    // materializing this view five times is the dominant cost of tall Muon's
    // Newton-Schulz loop.  The Lt path still receives a dense copy below.
    Tensor self_contig = self.is_contiguous()
        ? self
        : (native_cublas_dtype && self_transposed_contiguous
               ? self
               : self.contiguous());

    // The decoder linear layer pattern ``x @ weight.t()``: keep the weight
    // view untouched (its memory already reads as the transposed operand).
    const auto other_strides = other.strides();
    const bool transposed_contiguous =
        bias == nullptr && !other.is_contiguous() && other.dim() == 2 &&
        other_strides[0] == 1 && other_strides[1] == self_contig.shape()[1];

    Tensor other_contig;
    const void* a_ptr = nullptr;   // "other" slot of the GEMM
    const void* b_ptr = nullptr;   // "self" slot of the GEMM
    bool other_transposed = false;
    if (transposed_contiguous) {
        other_transposed = true;
        a_ptr = other.data_ptr();
        b_ptr = self_contig.data_ptr();
    } else {
        other_contig = other.is_contiguous() ? other : other.contiguous();
        a_ptr = other_contig.data_ptr();
        b_ptr = self_contig.data_ptr();
    }

    int64_t M = self_contig.shape()[0];
    int64_t K = self_contig.shape()[1];
    int64_t N = transposed_contiguous ? other.shape()[1] : other_contig.shape()[1];

    const DType dtype = self_contig.dtype();
    if (isComplexType(dtype)) {
        // cublasLt's complex algorithm set is not available uniformly across
        // CUDA versions.  The classic GEMM API has the same row-major
        // transpose trick and is the stable path for complex matmul.
        const cudaDataType_t cuda_type = to_cublas_type(dtype);
        const cublasComputeType_t compute_type = to_compute_type(dtype);
        void* alpha_ptr = to_scalar_ptr(alpha, dtype, 0);
        void* beta_ptr = to_scalar_ptr(beta, dtype, 1);
        const cublasOperation_t trans_a =
            other_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t trans_b =
            self_transposed_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = static_cast<int>(other_transposed ? K : N);
        const int ldb = static_cast<int>(self_transposed_contiguous ? M : K);
        CUBLAS_CHECK(cublasGemmEx(
            CUDAContext::getCublasHandle(),
            trans_a, trans_b,
            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
            alpha_ptr,
            a_ptr, cuda_type, lda,
            b_ptr, cuda_type, ldb,
            beta_ptr,
            result.data_ptr(), cuda_type, static_cast<int>(N),
            compute_type, CUBLAS_GEMM_DEFAULT));
        return;
    }

    // Match the native PyTorch default on this host: its preferred backend
    // is cuBLAS (not cuBLASLt) for Half/BFloat16 GEMM.  Besides avoiding Lt's
    // per-shape plan/autotune cost, this matters for tall Newton-Schulz
    // products where the cuBLAS reduction policy is the reference numerical
    // path.  Keep the Lt bias epilogue for vector-bias addmm below.
    const bool has_bias = (bias != nullptr);
    if (!has_bias && (dtype == DType::Float16 || dtype == DType::BFloat16)) {
        const cudaDataType_t cuda_type = to_cublas_type(dtype);
        const cublasComputeType_t compute_type = to_compute_type(dtype);
        void* alpha_ptr = to_scalar_ptr(alpha, dtype, 0);
        void* beta_ptr = to_scalar_ptr(beta, dtype, 1);
        cublasHandle_t handle = CUDAContext::getCublasHandle();
        // In the transposed-weights case `a_ptr` points at the underlying
        // contiguous [N,K] row-major storage while the logical operand is
        // its [K,N] transpose.  The row-major transpose trick therefore
        // needs OP_T and the underlying column-major leading dimension K.
        // OP_N/lda=N happens to work for some square/aligned shapes but
        // silently reads the wrong reduction order for Muon's live .T view.
        const cublasOperation_t trans_a =
            other_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t trans_b =
            self_transposed_contiguous ? CUBLAS_OP_T : CUBLAS_OP_N;
        const int lda = static_cast<int>(other_transposed ? K : N);
        const cublasStatus_t status = cublasGemmEx(
            handle,
            trans_a, trans_b,
            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
            alpha_ptr,
            a_ptr, cuda_type, lda,
            b_ptr, cuda_type,
            static_cast<int>(self_transposed_contiguous ? M : K),
            beta_ptr,
            result.data_ptr(), cuda_type, static_cast<int>(N),
            compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        CUBLAS_CHECK(status);
        return;
    }

    auto plan = get_gemm_plan(dtype, M, N, K, has_bias, other_transposed);
    std::lock_guard<std::mutex> execution_lock(plan->execution_mutex);

    void* alpha_ptr = to_scalar_ptr(alpha, dtype, 0);
    // Autotune trials must not read/accumulate into C, so they run with
    // beta = 0; the caller's beta is written into the same scale slot after
    // tuning and used for the final execution below.
    void* beta_ptr = to_scalar_ptr(0.0, dtype, 1);

    // Bias pointer for the bias epilogue (CUDA 11-style API: desc attribute).
    if (has_bias) {
        void* bias_ptr = bias->data_ptr();
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                       &bias_ptr, sizeof(void*)));
    }

    const bool run_autotune = !plan->autotuned && !tensorplay::cuda::isCapturing();

    // Autotune trials write D in place.  Preserve the caller's C only while
    // those trials run and beta is non-zero, so the real beta*C + alpha*A*B
    // execution below still sees the original accumulation buffer.  Without
    // this, the trials' beta=0 output is fed into the final beta=1 call and
    // the first GEMM using a broadcast addmm bias returns (roughly) 2*AB.
    Tensor saved_result;
    if (run_autotune && beta != 0.0) {
        saved_result = result.clone();
    }

    // One-time micro-autotune: time every heuristic candidate and keep the
    // measured winner at index 0.  cuBLASLt's top-1 estimate is not always
    // the fastest kernel for a given arch; this makes the choice empirical
    // while paying the cost only on the first call per (shape, dtype) key.
    //
    // Skipped while a CUDA graph capture is live: the trials record events
    // on the capturing stream, which aborts capture.  The plan is then
    // PINNED to heuristic candidate 0 so captured and eager executions use
    // bit-identical algorithms (mixing algorithms shows up as ulp-level
    // divergence between replay and eager recompute).
    if (run_autotune) {
        int best = 0;
        float best_ms = std::numeric_limits<float>::max();
        for (size_t c = 0; c < plan->candidates.size(); ++c) {
            cudaEvent_t ev_start, ev_end;
            cudaEventCreate(&ev_start);
            cudaEventCreate(&ev_end);
            // Warm up this algorithm once (also validates it really runs).
            cublasStatus_t st = cublasLtMatmul(
                CUDAContext::getCublasLtHandle(), plan->matmul_desc, alpha_ptr,
                a_ptr, plan->a_desc,
                b_ptr, plan->b_desc,
                beta_ptr,
                result.data_ptr(), plan->c_desc,
                result.data_ptr(), plan->c_desc,
                &plan->candidates[c].algo, plan->workspace_ptr, plan->workspace_size,
                getCurrentCUDAStream().stream());
            if (st != CUBLAS_STATUS_SUCCESS) {
                cudaEventDestroy(ev_start);
                cudaEventDestroy(ev_end);
                continue;
            }
            constexpr int kTrials = 3;
            cudaEventRecord(ev_start, getCurrentCUDAStream().stream());
            for (int t = 0; t < kTrials; ++t) {
                cublasLtMatmul(
                    CUDAContext::getCublasLtHandle(), plan->matmul_desc, alpha_ptr,
                    a_ptr, plan->a_desc,
                    b_ptr, plan->b_desc,
                    beta_ptr,
                    result.data_ptr(), plan->c_desc,
                    result.data_ptr(), plan->c_desc,
                    &plan->candidates[c].algo, plan->workspace_ptr, plan->workspace_size,
                    getCurrentCUDAStream().stream());
            }
            cudaEventRecord(ev_end, getCurrentCUDAStream().stream());
            cudaEventSynchronize(ev_end);
            float ms = 0;
            cudaEventElapsedTime(&ms, ev_start, ev_end);
            cudaEventDestroy(ev_start);
            cudaEventDestroy(ev_end);
            if (ms < best_ms) {
                best_ms = ms;
                best = static_cast<int>(c);
            }
        }
        if (best != 0) {
            std::swap(plan->candidates[0], plan->candidates[static_cast<size_t>(best)]);
        }
        plan->autotuned = true;
    } else if (!plan->autotuned) {
        // First use happened under capture: pin heuristic candidate 0.
        plan->autotuned = true;
    }

    // Restore the caller's beta for the real execution.
    beta_ptr = to_scalar_ptr(beta, dtype, 1);

    if (run_autotune && beta != 0.0) {
        result.copy_(saved_result);
    }

    CUBLASLT_CHECK(cublasLtMatmul(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, alpha_ptr,
        a_ptr, plan->a_desc,
        b_ptr, plan->b_desc,
        beta_ptr,
        result.data_ptr(), plan->c_desc,
        result.data_ptr(), plan->c_desc,
        &plan->candidates[0].algo, plan->workspace_ptr, plan->workspace_size,
        getCurrentCUDAStream().stream()));
}

void gemm_strided_batched_3d(const Tensor& self_3d, const Tensor& other_3d,
                             Tensor& result_3d, int64_t batch_size,
                             int64_t M, int64_t N, int64_t K,
                             long long stride_a, long long stride_b,
                             double alpha, double beta) {
    if (batch_size == 0 || M == 0 || N == 0) return;
    if (K == 0) {
        zero_matmul_output(result_3d);
        return;
    }
    const DType dtype = self_3d.dtype();
    const cudaDataType_t cuda_type = to_cublas_type(dtype);
    const cublasComputeType_t compute_type = to_compute_type(dtype);
    void* alpha_ptr = to_scalar_ptr(alpha, dtype, 0);
    void* beta_ptr = to_scalar_ptr(beta, dtype, 1);
    const long long stride_c = static_cast<long long>(M) * N;
    // Torch leaves the algorithm choice to cuBLAS defaults; the TENSOR_OP
    // hint only affects kernel selection, never the FP32-accumulate contract.
    const cublasGemmAlgo_t algorithm = isComplexType(dtype)
        ? CUBLAS_GEMM_DEFAULT
        : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        CUDAContext::getCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        alpha_ptr,
        other_3d.data_ptr(), cuda_type, static_cast<int>(N), stride_b,
        self_3d.data_ptr(), cuda_type, static_cast<int>(K), stride_a,
        beta_ptr,
        result_3d.data_ptr(), cuda_type, static_cast<int>(N), stride_c,
        static_cast<int>(batch_size), compute_type, algorithm));
}

} // namespace cuda
} // namespace tensorplay
