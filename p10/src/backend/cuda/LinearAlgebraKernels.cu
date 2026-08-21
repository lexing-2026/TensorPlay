#include "Tensor.h"
#include "TypePromotion.h"
#include "Dispatcher.h"
#include "CUDAContext.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Scalar.h"
#include "Utils.h"
#include "GradMode.h"
#include "LinearAlgebraNames.h"
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include <limits>
#include <string>
#include <vector>
#include <algorithm>

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

Tensor& zero_matmul_output_cuda(Tensor& output) {
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
// So: opA = opB = N, A pointer = B_r, B pointer = A_r, no transposes needed.
struct GemmKey {
    ScalarType dtype;
    int64_t m, n, k;
    int device;
    bool has_bias;

    bool operator==(const GemmKey& o) const {
        return dtype == o.dtype && m == o.m && n == o.n && k == o.k &&
               device == o.device && has_bias == o.has_bias;
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
    cublasLtMatmulHeuristicResult_t heuristic{};
    bool heuristic_valid = false;
    size_t workspace_size = 0;

    ~GemmPlan() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (matmul_desc) cublasLtMatmulDescDestroy(matmul_desc);
    }
};

std::mutex& planMutex() {
    static auto* mutex = new std::mutex();
    return *mutex;
}

std::unordered_map<GemmKey, std::shared_ptr<GemmPlan>, GemmKeyHash>& planCache() {
    // Plans own device workspaces and intentionally have process lifetime.
    static auto* cache = new std::unordered_map<GemmKey, std::shared_ptr<GemmPlan>, GemmKeyHash>();
    return *cache;
}

cudaDataType_t toCublasType(ScalarType t) {
    switch (t) {
        case ScalarType::Float32: return CUDA_R_32F;
        case ScalarType::Float64: return CUDA_R_64F;
        case ScalarType::Float16: return CUDA_R_16F;
        case ScalarType::BFloat16: return CUDA_R_16BF;
        case ScalarType::ComplexFloat: return CUDA_C_32F;
        case ScalarType::ComplexDouble: return CUDA_C_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

cublasComputeType_t toComputeType(ScalarType t) {
    switch (t) {
        // PyTorch's default is torch.backends.cuda.matmul.allow_tf32=False.
        // Use ordinary FP32 accumulation so matmul has the same numerical
        // contract instead of silently enabling TF32.
        case ScalarType::Float32: return CUBLAS_COMPUTE_32F;
        case ScalarType::Float64: return CUBLAS_COMPUTE_64F;
        // PyTorch accumulates Half GEMM in FP32 by default
        // (torch.backends.cuda.matmul.allow_fp16_accumulation is False), and
        // its CUDABlas helpers pass CUDA_R_32F as the compute type with float
        // alpha/beta for both Half and BFloat16.
        case ScalarType::Float16: return CUBLAS_COMPUTE_32F;
        case ScalarType::BFloat16: return CUBLAS_COMPUTE_32F;
        case ScalarType::ComplexFloat: return CUBLAS_COMPUTE_32F;
        case ScalarType::ComplexDouble: return CUBLAS_COMPUTE_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

cudaDataType_t toScaleType(ScalarType t) {
    switch (t) {
        case ScalarType::Float32: return CUDA_R_32F;
        case ScalarType::Float64: return CUDA_R_64F;
        // FP32 compute for reduced-precision inputs: scale in FP32.
        case ScalarType::Float16: return CUDA_R_32F;
        // Compute is FP32-based (CUBLAS_COMPUTE_32F): scale in FP32.
        case ScalarType::BFloat16: return CUDA_R_32F;
        case ScalarType::ComplexFloat: return CUDA_C_32F;
        case ScalarType::ComplexDouble: return CUDA_C_64F;
        default: TP_THROW(NotImplementedError, "mm: unsupported dtype on CUDA");
    }
}

// Alpha/beta must live in the scale type of the compute type.
void* toScalarPtr(double v, ScalarType t, int slot) {
    static thread_local float f32[2];
    static thread_local double f64[2];
    static thread_local cuFloatComplex c32[2];
    static thread_local cuDoubleComplex c64[2];
    switch (t) {
        case ScalarType::Float32: f32[slot] = static_cast<float>(v); return &f32[slot];
        case ScalarType::Float64: f64[slot] = v; return &f64[slot];
        // FP32 compute for Half/BFloat16: alpha/beta live in the float scale type.
        case ScalarType::Float16: f32[slot] = static_cast<float>(v); return &f32[slot];
        case ScalarType::BFloat16: f32[slot] = static_cast<float>(v); return &f32[slot];
        case ScalarType::ComplexFloat: c32[slot] = make_cuFloatComplex(static_cast<float>(v), 0.0f); return &c32[slot];
        case ScalarType::ComplexDouble: c64[slot] = make_cuDoubleComplex(v, 0.0); return &c64[slot];
        default: return nullptr;
    }
}

std::shared_ptr<GemmPlan> getGemmPlan(ScalarType dtype, int64_t M, int64_t N, int64_t K, bool has_bias) {
    const int device = currentDevice();
    GemmKey key{dtype, M, N, K, device, has_bias};
    std::lock_guard<std::mutex> lock(planMutex());
    auto& cache = planCache();
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;

    auto plan = std::make_shared<GemmPlan>();
    plan->device = device;
    cudaDataType_t cudaType = toCublasType(dtype);
    cublasComputeType_t computeType = toComputeType(dtype);
    cudaDataType_t scaleType = toScaleType(dtype);

    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&plan->matmul_desc, computeType, scaleType));
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtEpilogue_t epilogue = has_bias ? CUBLASLT_EPILOGUE_BIAS : CUBLASLT_EPILOGUE_DEFAULT;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));

    // A' = B_r (N x K) col-major ld=N; B' = A_r (K x M) col-major ld=K; C' (N x M) ld=N.
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->a_desc, cudaType, N, K, N));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->b_desc, cudaType, K, M, K));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&plan->c_desc, cudaType, N, M, N));

    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&plan->pref));
    size_t workspace_size = 32 * 1024 * 1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(plan->pref,
                                                         CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                         &workspace_size, sizeof(workspace_size)));
    plan->workspace_size = workspace_size;

    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, plan->a_desc, plan->b_desc, plan->c_desc, plan->c_desc,
        plan->pref, 1, &plan->heuristic, &returned));
    if (returned == 0) {
        TP_THROW(RuntimeError, "cuBLASLt: no heuristic algorithm found");
    }
    plan->heuristic_valid = true;

    cache.emplace(key, plan);
    return plan;
}

void gemm_impl(const Tensor& self, const Tensor& other, const Tensor& result,
               double alpha, double beta, const Tensor* bias) {
    if (self.dim() != 2 || other.dim() != 2) {
        TP_THROW(RuntimeError, "mm: tensors must be 2D");
    }
    if (self.shape()[1] != other.shape()[0]) {
        TP_THROW(RuntimeError, "mm: shape mismatch");
    }

    Tensor self_contig = self.is_contiguous() ? self : self.contiguous();

    // A decoder linear layer normally calls ``x @ weight.t()``.  Materializing
    // that transpose here would transiently duplicate a 50--135 MiB weight
    // matrix on every projection and is the dominant peak-memory gap versus
    // Torch.  A contiguous [N, K] weight viewed as [K, N] has strides [1, K];
    // map that view directly to column-major cuBLAS (W * X^T) instead.
    const auto other_strides = other.strides();
    const bool transposed_contiguous =
        bias == nullptr && !other.is_contiguous() && other.dim() == 2 &&
        other_strides[0] == 1 && other_strides[1] == self_contig.shape()[1];
    if (transposed_contiguous) {
        const int64_t M = self_contig.shape()[0];
        const int64_t K = self_contig.shape()[1];
        const int64_t N = other.shape()[1];
        const ScalarType dtype = self_contig.dtype();
        const cudaDataType_t cuda_type = toCublasType(dtype);
        const cublasComputeType_t compute_type = toComputeType(dtype);
        void* alpha_ptr = toScalarPtr(alpha, dtype, 0);
        void* beta_ptr = toScalarPtr(beta, dtype, 1);
        CUBLAS_CHECK(cublasGemmEx(
            CUDAContext::getCublasHandle(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
            alpha_ptr,
            other.data_ptr(), cuda_type, static_cast<int>(K),
            self_contig.data_ptr(), cuda_type, static_cast<int>(K),
            beta_ptr,
            result.data_ptr(), cuda_type, static_cast<int>(N),
        compute_type, CUBLAS_GEMM_DEFAULT));
        return;
    }

    Tensor other_contig = other.is_contiguous() ? other : other.contiguous();

    int64_t M = self_contig.shape()[0];
    int64_t K = self_contig.shape()[1];
    int64_t N = other_contig.shape()[1];

    const ScalarType dtype = self_contig.dtype();
    if (isComplexType(dtype)) {
        // cublasLt's complex algorithm set is not available uniformly across
        // CUDA versions.  The classic GEMM API has the same row-major
        // transpose trick and is the stable path for complex matmul.
        const cudaDataType_t cuda_type = toCublasType(dtype);
        const cublasComputeType_t compute_type = toComputeType(dtype);
        void* alpha_ptr = toScalarPtr(alpha, dtype, 0);
        void* beta_ptr = toScalarPtr(beta, dtype, 1);
        CUBLAS_CHECK(cublasGemmEx(
            CUDAContext::getCublasHandle(),
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
            alpha_ptr,
            other_contig.data_ptr(), cuda_type, static_cast<int>(N),
            self_contig.data_ptr(), cuda_type, static_cast<int>(K),
            beta_ptr,
            result.data_ptr(), cuda_type, static_cast<int>(N),
            compute_type, CUBLAS_GEMM_DEFAULT));
        return;
    }
    bool has_bias = (bias != nullptr);
    auto plan = getGemmPlan(dtype, M, N, K, has_bias);
    Tensor workspace({static_cast<int64_t>(plan->workspace_size)}, DType::UInt8, self.device());
    std::lock_guard<std::mutex> execution_lock(plan->execution_mutex);

    void* alpha_ptr = toScalarPtr(alpha, dtype, 0);
    void* beta_ptr = toScalarPtr(beta, dtype, 1);

    // Bias pointer for the bias epilogue (CUDA 11-style API: set as desc attribute).
    if (has_bias) {
        void* bias_ptr = bias->data_ptr();
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                       &bias_ptr, sizeof(void*)));
    }

    cublasStatus_t status = cublasLtMatmul(
        CUDAContext::getCublasLtHandle(), plan->matmul_desc, alpha_ptr,
        other_contig.data_ptr(), plan->a_desc,
        self_contig.data_ptr(), plan->b_desc,
        beta_ptr,
        result.data_ptr(), plan->c_desc,
        result.data_ptr(), plan->c_desc,
        &plan->heuristic.algo, workspace.data_ptr(), plan->workspace_size,
        getCurrentCUDAStream().stream());
    CUBLASLT_CHECK(status);
}

bool is_bias_vector(const Tensor& b, int64_t M, int64_t N) {
    if (b.dim() == 1 && b.shape()[0] == N) return true;
    if (b.dim() == 2 && b.shape()[0] == 1 && b.shape()[1] == N) return true;
    return false;
}

// Torch broadcasts `input` against the GEMM output shape right-aligned and
// reports mismatches through its expand wording.  Returns a (possibly
// zero-stride) view; callers materialize with clone()/contiguous() before
// mutating.  Mirrors expand_gemm_input in the CPU kernel.
Tensor expand_gemm_input_cuda(const Tensor& input, const std::vector<int64_t>& target) {
    const int64_t td = target.size();
    const int64_t sd = input.dim();
    if (sd > td) {
        TP_THROW(RuntimeError, "expand: the number of sizes provided (", td,
                 ") must be greater or equal to the number of dimensions in the tensor (", sd, ")");
    }

    std::vector<int64_t> src(td, 1);
    std::vector<int64_t> src_strides(td, 0);
    for (int64_t i = 0; i < sd; ++i) {
        src[td - 1 - i] = input.size(sd - 1 - i);
        src_strides[td - 1 - i] = input.stride(sd - 1 - i);
    }

    for (int64_t k = td - 1; k >= 0; --k) {
        if (src[k] != 1 && src[k] != target[k]) {
            std::string tgt = "[";
            std::string own = "[";
            for (int64_t d = 0; d < td; ++d) {
                if (d) { tgt += ", "; own += ", "; }
                tgt += std::to_string(target[d]);
                own += std::to_string(src[d]);
            }
            tgt += "]";
            own += "]";
            TP_THROW(RuntimeError, "The expanded size of the tensor (", target[k],
                     ") must match the existing size (", src[k], ") at non-singleton dimension ",
                     k, ".  Target sizes: ", tgt, ".  Tensor sizes: ", own);
        }
    }

    std::vector<int64_t> out_strides(td);
    for (int64_t k = 0; k < td; ++k) {
        out_strides[k] = (src[k] == 1 && target[k] != 1) ? 0 : src_strides[k];
    }
    return input.as_strided(target, out_strides);
}

} // anonymous namespace

Tensor mm_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (other.dim() != 2) TP_THROW(RuntimeError, "mat2 must be a matrix");
    if (self.size(1) != other.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", self.size(0),
                 "x", self.size(1), " and ", other.size(0), "x", other.size(1), ")");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(other.dtype()));
    }
    const DType result_dtype = self.dtype();
    // Validate the dtype before any empty/K==0 fast path.  Torch rejects
    // integer and bool CUDA matmul even when the mathematical result is empty
    // or identically zero.
    (void)toCublasType(result_dtype);
    const Tensor& self_p = self;
    const Tensor& other_p = other;
    int64_t M = self_p.shape()[0];
    int64_t N = other_p.shape()[1];
    Tensor result = Tensor::empty({M, N}, result_dtype, self.device());
    if (M == 0 || N == 0) {
        return result;
    }
    if (self_p.size(1) == 0) {
        return zero_matmul_output_cuda(result);
    }
    gemm_impl(self_p, other_p, result, 1.0, 0.0, nullptr);
    return result;
}

std::vector<int64_t> decode_batch_index_cuda(
    int64_t linear_index, const std::vector<int64_t>& batch_shape) {
    std::vector<int64_t> index(batch_shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(batch_shape.size()) - 1; dim >= 0; --dim) {
        const int64_t size = batch_shape[dim];
        index[dim] = size == 0 ? 0 : linear_index % size;
        if (size != 0) linear_index /= size;
    }
    return index;
}

Tensor select_batch_matrix_cuda(
    const Tensor& input,
    const std::vector<int64_t>& input_batch_shape,
    const std::vector<int64_t>& output_batch_shape,
    const std::vector<int64_t>& output_index) {
    Tensor matrix = input;
    const int64_t padding = static_cast<int64_t>(output_batch_shape.size()) -
                            static_cast<int64_t>(input_batch_shape.size());
    for (int64_t dim = static_cast<int64_t>(input_batch_shape.size()) - 1; dim >= 0; --dim) {
        const int64_t output_dim = padding + dim;
        const int64_t index = input_batch_shape[dim] == 1 ? 0 : output_index[output_dim];
        matrix = matrix.select(dim, index);
    }
    return matrix;
}

Tensor select_output_matrix_cuda(
    const Tensor& output, const std::vector<int64_t>& output_index) {
    Tensor matrix = output;
    for (int64_t dim = static_cast<int64_t>(output_index.size()) - 1; dim >= 0; --dim) {
        matrix = matrix.select(dim, output_index[dim]);
    }
    return matrix;
}

Tensor materialize_batched_cuda(
    const Tensor& input,
    const std::vector<int64_t>& input_batch_shape,
    const std::vector<int64_t>& output_batch_shape,
    int64_t rows, int64_t cols, int64_t batch_size) {
    const bool all_singleton_batch_dims = std::all_of(
        input_batch_shape.begin(), input_batch_shape.end(),
        [](int64_t size) { return size == 1; });

    // A matrix (or an all-singleton batch) can be consumed directly with a
    // zero batch stride.  Expanding and cloning it here adds a full broadcast
    // copy to every batched matmul, while cuBLAS already supports stride==0.
    // Only materialize when the matrix itself is not row-major contiguous.
    const int64_t matrix_dim = input.dim();
    const bool matrix_contiguous =
        matrix_dim >= 2 && input.stride(matrix_dim - 1) == 1 &&
        input.stride(matrix_dim - 2) == cols;
    if (all_singleton_batch_dims) {
        return matrix_contiguous ? input : input.contiguous();
    }

    std::vector<int64_t> target_shape = output_batch_shape;
    target_shape.push_back(rows);
    target_shape.push_back(cols);
    Tensor normalized = input;
    if (input_batch_shape != output_batch_shape ||
        input.dim() != static_cast<int64_t>(output_batch_shape.size()) + 2) {
        normalized = input.expand(target_shape);
    }
    // cublasGemmStridedBatchedEx requires regular strides.  This clone also
    // materializes zero-stride broadcast dimensions and arbitrary matrix
    // views in one pass.
    if (!normalized.is_contiguous()) normalized = normalized.clone();
    return normalized.reshape({batch_size, rows, cols});
}

long long batched_matrix_stride_cuda(
    const std::vector<int64_t>& input_batch_shape,
    const std::vector<int64_t>& output_batch_shape,
    int64_t rows, int64_t cols) {
    const bool all_singleton_batch_dims = std::all_of(
        input_batch_shape.begin(), input_batch_shape.end(),
        [](int64_t size) { return size == 1; });
    if (input_batch_shape != output_batch_shape && all_singleton_batch_dims) {
        return 0;
    }
    return static_cast<long long>(rows) * static_cast<long long>(cols);
}

// Launch one strided-batched GEMM over contiguous (B, M, K) x (B, K, N)
// operand stacks into a contiguous (B, M, N) result.
void gemm_strided_batched_3d(const Tensor& self_3d, const Tensor& other_3d,
                             Tensor& result_3d, int64_t batch_size,
                             int64_t M, int64_t N, int64_t K,
                             double alpha, double beta) {
    const ScalarType dtype = self_3d.dtype();
    const cudaDataType_t cuda_type = toCublasType(dtype);
    const cublasComputeType_t compute_type = toComputeType(dtype);
    void* alpha_ptr = toScalarPtr(alpha, dtype, 0);
    void* beta_ptr = toScalarPtr(beta, dtype, 1);
    const long long stride_a = static_cast<long long>(M) * K;
    const long long stride_b = static_cast<long long>(K) * N;
    const long long stride_c = static_cast<long long>(M) * N;
    // Torch leaves the algorithm choice to cuBLAS defaults; the TENSOR_OP
    // hint only affects kernel selection, never the FP32-accumulate contract.
    const cublasGemmAlgo_t algorithm = isComplexType(dtype)
        ? CUBLAS_GEMM_DEFAULT
        : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    if (batch_size == 0 || M == 0 || N == 0) return;
    if (K == 0) {
        zero_matmul_output_cuda(result_3d);
        return;
    }
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

Tensor matmul_batched_2d_cuda(
    const Tensor& self, const Tensor& other,
    const std::vector<int64_t>& self_batch_shape,
    const std::vector<int64_t>& other_batch_shape) {
    if (self.dim() < 2 || other.dim() < 2) {
        TP_THROW(RuntimeError, "matmul: internal operands must be at least 2D");
    }
    const int64_t M = self.size(self.dim() - 2);
    const int64_t K = self.size(self.dim() - 1);
    if (K != other.size(other.dim() - 2)) {
        TP_THROW(RuntimeError, "matmul: size mismatch, got ", K, " and ", other.size(other.dim() - 2));
    }
    (void)toCublasType(self.dtype());
    const int64_t N = other.size(other.dim() - 1);
    const std::vector<int64_t> batch_shape = broadcast_shapes(self_batch_shape, other_batch_shape);

    std::vector<int64_t> result_shape = batch_shape;
    result_shape.push_back(M);
    result_shape.push_back(N);
    Tensor result = Tensor::empty(result_shape, self.dtype(), self.device());

    int64_t batch_size = 1;
    for (const int64_t size : batch_shape) {
        if (size == 0) return result;
        if (batch_size > std::numeric_limits<int64_t>::max() / size) {
            TP_THROW(RuntimeError, "matmul: output batch size overflow");
        }
        batch_size *= size;
    }

    if (M == 0 || N == 0) {
        return result;
    }

    if (K == 0) {
        return zero_matmul_output_cuda(result);
    }

    if (batch_shape.empty()) {
        return mm_kernel_cuda(self, other);
    }

    // Torch lowers ``(batch..., M, K) @ (K, N)`` to one large 2-D mm by
    // folding the batch into M.  Besides matching that execution contract,
    // this avoids launching a strided-batched kernel for the common linear
    // layer pattern with one shared weight matrix.
    if (other_batch_shape.empty()) {
        Tensor self_materialized = materialize_batched_cuda(
            self, self_batch_shape, batch_shape, M, K, batch_size);
        Tensor self_2d = self_materialized.reshape({batch_size * M, K});
        Tensor result_2d = result.reshape({batch_size * M, N});
        gemm_impl(self_2d, other, result_2d, 1.0, 0.0, nullptr);
        return result;
    }

    Tensor self_3d = materialize_batched_cuda(
        self, self_batch_shape, batch_shape, M, K, batch_size);
    Tensor other_3d = materialize_batched_cuda(
        other, other_batch_shape, batch_shape, K, N, batch_size);
    Tensor result_3d = result.reshape({batch_size, M, N});

    const ScalarType dtype = self.dtype();
    const cudaDataType_t cuda_type = toCublasType(dtype);
    const cublasComputeType_t compute_type = toComputeType(dtype);
    void* alpha_ptr = toScalarPtr(1.0, dtype, 0);
    void* beta_ptr = toScalarPtr(0.0, dtype, 1);
    const long long stride_a = batched_matrix_stride_cuda(
        other_batch_shape, batch_shape, K, N);
    const long long stride_b = batched_matrix_stride_cuda(
        self_batch_shape, batch_shape, M, K);
    const long long stride_c = static_cast<long long>(M) * N;
    const cublasGemmAlgo_t batch_algorithm = isComplexType(dtype)
        ? CUBLAS_GEMM_DEFAULT
        : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        CUDAContext::getCublasHandle(),
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(N), static_cast<int>(M), static_cast<int>(K),
        alpha_ptr,
        other_3d.data_ptr(), cuda_type, static_cast<int>(N), stride_a,
        self_3d.data_ptr(), cuda_type, static_cast<int>(K), stride_b,
        beta_ptr,
        result_3d.data_ptr(), cuda_type, static_cast<int>(N), stride_c,
        static_cast<int>(batch_size), compute_type, batch_algorithm));
    return result;
}

Tensor addmm_kernel_cuda(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                         Scalar beta, Scalar alpha) {
    if (mat1.dim() != 2 || mat2.dim() != 2) TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (",
        mat1.dim(), "D and ", mat2.dim(), "D)");
    if (mat1.size(1) != mat2.size(0)) {
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", mat1.size(0), "x", mat1.size(1),
                 " and ", mat2.size(0), "x", mat2.size(1), ")");
    }
    // Torch requires all three operands to share one dtype; it checks
    // self-vs-mat2 first, then mat1-vs-mat2 (LinearAlgebra.cpp:185-186).
    if (input.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "self and mat2 must have the same dtype, but got ",
                 pretty_dtype_name(input.dtype()), " and ", pretty_dtype_name(mat2.dtype()));
    }
    if (mat1.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "mat1 and mat2 must have the same dtype, but got ",
                 pretty_dtype_name(mat1.dtype()), " and ", pretty_dtype_name(mat2.dtype()));
    }
    int64_t M = mat1.size(0);
    int64_t N = mat2.size(1);
    double alpha_v = alpha.toDouble();
    double beta_v = beta.toDouble();

    // out = beta * input + alpha * (self @ other)
    if (beta_v == 0.0) {
        Tensor result = Tensor::empty({M, N}, mat1.dtype(), mat1.device());
        gemm_impl(mat1, mat2, result, alpha_v, 0.0, nullptr);
        return result;
    }

    // Broadcast like torch: any input broadcastable to (M, N) is accepted.
    // The cuBLASLt bias epilogue stays as the fast path for the common
    // vector/(1,N) bias with beta == 1.
    if (beta_v == 1.0 && is_bias_vector(input, M, N)) {
        Tensor result = Tensor::empty({M, N}, mat1.dtype(), mat1.device());
        gemm_impl(mat1, mat2, result, alpha_v, 0.0, &input);
        return result;
    }

    Tensor result = expand_gemm_input_cuda(input, {M, N}).clone();
    gemm_impl(mat1, mat2, result, alpha_v, beta_v, nullptr);
    return result;
}

Tensor matmul_kernel_cuda(const Tensor& self, const Tensor& other) {
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();
    if (dim1 < 1 || dim2 < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }

    // Shape contract first, using torch's exact wording (mirrors the CPU
    // kernel; see the comment there for the folding rules).
    const auto self_shape = static_cast<std::vector<int64_t>>(self.shape());
    const auto other_shape = static_cast<std::vector<int64_t>>(other.shape());

    if (dim1 == 1 && dim2 == 1) {
        if (self.size(0) != other.size(0)) {
            TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [", self.size(0),
                     "] and src [", other.size(0),
                     "] to have the same number of elements, but got ", self.size(0), " and ",
                     other.size(0), " elements respective");
        }
    } else if (dim1 == 1) {
        const int64_t k = self.size(0);
        const int64_t other_k = other_shape[other_shape.size() - 2];
        if (k != other_k) {
            TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (1x", k,
                     " and ", other_k, "x", other_shape.back(), ")");
        }
    } else if (dim2 == 1) {
        int64_t folded_m = self_shape[self_shape.size() - 2];
        for (size_t i = 0; i + 2 < self_shape.size(); ++i) folded_m *= self_shape[i];
        const int64_t k = self_shape.back();
        if (k != other.size(0)) {
            TP_THROW(RuntimeError, "size mismatch, got input (", folded_m, "), mat (", folded_m,
                     "x", k, "), vec (", other.size(0), ")");
        }
    } else {
        const int64_t k = self_shape.back();
        const int64_t other_k = other_shape[other_shape.size() - 2];
        if (k != other_k) {
            if (dim2 == 2) {
                int64_t folded_m = self_shape[self_shape.size() - 2];
                for (size_t i = 0; i + 2 < self_shape.size(); ++i) folded_m *= self_shape[i];
                TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", folded_m,
                         "x", k, " and ", other_k, "x", other_shape.back(), ")");
            } else {
                std::vector<int64_t> self_batch(self_shape.begin(), self_shape.end() - 2);
                std::vector<int64_t> other_batch(other_shape.begin(), other_shape.end() - 2);
                const std::vector<int64_t> batch = broadcast_shapes(self_batch, other_batch);
                int64_t prod_batch = 1;
                for (const int64_t s : batch) prod_batch *= s;
                TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                         prod_batch, ", ", k, "] but got: [", prod_batch, ", ", other_k, "].");
            }
        }
    }

    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(other.dtype()));
    }
    const Tensor& self_p = self;
    const Tensor& other_p = other;

    if (dim1 == 1 && dim2 == 1) {
        if (self_p.size(0) != other_p.size(0)) {
            TP_THROW(RuntimeError, "matmul: size mismatch, got ", self_p.size(0), " and ", other_p.size(0));
        }
        Tensor result = matmul_batched_2d_cuda(
            self_p.unsqueeze(0), other_p.unsqueeze(1), {}, {});
        return result.squeeze(0).squeeze(0);
    }

    if (dim1 == 1) {
        const auto other_shape = static_cast<std::vector<int64_t>>(other_p.shape());
        std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
        Tensor result = matmul_batched_2d_cuda(
            self_p.unsqueeze(0), other_p, {}, other_batch_shape);
        return result.squeeze(-2);
    }

    if (dim2 == 1) {
        const auto self_shape = static_cast<std::vector<int64_t>>(self_p.shape());
        std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
        Tensor result = matmul_batched_2d_cuda(
            self_p, other_p.unsqueeze(-1), self_batch_shape, {});
        return result.squeeze(-1);
    }

    std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
    std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
    return matmul_batched_2d_cuda(self_p, other_p, self_batch_shape, other_batch_shape);
}

Tensor& matmul_out_kernel_cuda(const Tensor& self, const Tensor& other, Tensor& out) {
    if (out.device() != self.device()) {
        TP_THROW(DeviceMismatchError, "matmul: out tensor must be on the same device as the inputs");
    }
    if (out.dtype() != self.dtype()) {
        TP_THROW(RuntimeError, "Expected out tensor to have dtype ",
                 static_cast<int>(self.dtype()), ", but got ",
                 static_cast<int>(out.dtype()));
    }
    if (GradMode::is_enabled() &&
        (self.requires_grad() || other.requires_grad() || out.requires_grad())) {
        TP_THROW(RuntimeError,
                 "matmul(): functions with out=... arguments don't support automatic differentiation, "
                 "but one of the arguments requires grad.");
    }

    // Compute before touching `out`: Torch permits `out` to alias either input.
    Tensor result = matmul_kernel_cuda(self, other);
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

Tensor transpose_last_two_view_cuda(const Tensor& input) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "matmul backward: expected a matrix operand");
    }
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(input.shape());
    std::vector<int64_t> strides = input.strides();
    std::swap(sizes[sizes.size() - 2], sizes[sizes.size() - 1]);
    std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
    return input.as_strided(sizes, strides);
}

__global__ void conjugate_complex_float_kernel(
    int64_t n, const cuFloatComplex* input, cuFloatComplex* output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) output[index] = cuConjf(input[index]);
}

__global__ void conjugate_complex_double_kernel(
    int64_t n, const cuDoubleComplex* input, cuDoubleComplex* output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n) output[index] = cuConj(input[index]);
}

Tensor adjoint_last_two_cuda(const Tensor& input) {
    Tensor transposed = transpose_last_two_view_cuda(input);
    if (input.dtype() != DType::ComplexFloat && input.dtype() != DType::ComplexDouble) {
        return transposed;
    }

    Tensor contiguous = transposed.contiguous();
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(contiguous.shape()), input.dtype(), input.device());
    const int64_t n = contiguous.numel();
    if (n == 0) return result;
    const dim3 block(256);
    const dim3 grid(static_cast<unsigned>((n + 255) / 256));
    if (input.dtype() == DType::ComplexFloat) {
        conjugate_complex_float_kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n,
            reinterpret_cast<const cuFloatComplex*>(contiguous.data_ptr<std::complex<float>>()),
            reinterpret_cast<cuFloatComplex*>(result.data_ptr<std::complex<float>>()));
    } else {
        conjugate_complex_double_kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
            n,
            reinterpret_cast<const cuDoubleComplex*>(contiguous.data_ptr<std::complex<double>>()),
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr<std::complex<double>>()));
    }
    checkCuda(cudaGetLastError(), "matmul conjugate transpose kernel launch");
    return result;
}

// Keep matmul's shape metadata in line with Torch's maximum tensor rank.
constexpr int kMatmulShapeMaxDims = 64;

struct MatmulSumShapeInfo {
    int source_ndim;
    int target_ndim;
    int64_t source_sizes[kMatmulShapeMaxDims];
    int64_t source_strides[kMatmulShapeMaxDims];
    int64_t target_sizes[kMatmulShapeMaxDims];
    int64_t target_strides[kMatmulShapeMaxDims];
};

__device__ void decode_matmul_sum_index(
    int64_t linear_index,
    const MatmulSumShapeInfo& info,
    int64_t& source_offset,
    int64_t& target_offset) {
    source_offset = 0;
    target_offset = 0;
    const int leading = info.source_ndim - info.target_ndim;
    for (int dim = info.source_ndim - 1; dim >= 0; --dim) {
        const int64_t coordinate = linear_index % info.source_sizes[dim];
        linear_index /= info.source_sizes[dim];
        source_offset += coordinate * info.source_strides[dim];
        const int target_dim = dim - leading;
        if (target_dim >= 0 && info.target_sizes[target_dim] != 1) {
            target_offset += coordinate * info.target_strides[target_dim];
        }
    }
}

__global__ void sum_complex_float_to_shape_kernel(
    int64_t numel,
    const cuFloatComplex* source,
    cuFloatComplex* target,
    MatmulSumShapeInfo info) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) return;
    int64_t source_offset = 0;
    int64_t target_offset = 0;
    decode_matmul_sum_index(index, info, source_offset, target_offset);
    const cuFloatComplex value = source[source_offset];
    atomicAdd(&target[target_offset].x, value.x);
    atomicAdd(&target[target_offset].y, value.y);
}

__global__ void sum_complex_double_to_shape_kernel(
    int64_t numel,
    const cuDoubleComplex* source,
    cuDoubleComplex* target,
    MatmulSumShapeInfo info) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) return;
    int64_t source_offset = 0;
    int64_t target_offset = 0;
    decode_matmul_sum_index(index, info, source_offset, target_offset);
    const cuDoubleComplex value = source[source_offset];
    atomicAdd(&target[target_offset].x, value.x);
    atomicAdd(&target[target_offset].y, value.y);
}

Tensor sum_to_shape_complex_cuda(
    const Tensor& input, const std::vector<int64_t>& target_shape) {
    const auto source_shape = static_cast<std::vector<int64_t>>(input.shape());
    if (source_shape.size() > kMatmulShapeMaxDims ||
        target_shape.size() > kMatmulShapeMaxDims) {
        TP_THROW(RuntimeError, "matmul backward: tensor rank exceeds CUDA limit");
    }

    MatmulSumShapeInfo info{};
    info.source_ndim = static_cast<int>(source_shape.size());
    info.target_ndim = static_cast<int>(target_shape.size());
    if (info.target_ndim > info.source_ndim) {
        TP_THROW(RuntimeError, "matmul backward: target rank exceeds source rank");
    }
    const int leading = info.source_ndim - info.target_ndim;
    for (int dim = 0; dim < info.source_ndim; ++dim) {
        info.source_sizes[dim] = source_shape[dim];
        info.source_strides[dim] = input.stride(dim);
    }
    for (int dim = 0; dim < info.target_ndim; ++dim) {
        const int64_t source_dim = source_shape[leading + dim];
        if (target_shape[dim] != 1 && target_shape[dim] != source_dim) {
            TP_THROW(RuntimeError, "matmul backward: incompatible gradient shape");
        }
        info.target_sizes[dim] = target_shape[dim];
    }

    Tensor result = Tensor::empty(target_shape, input.dtype(), input.device());
    zero_matmul_output_cuda(result);
    if (input.numel() == 0 || result.numel() == 0) return result;
    for (int dim = 0; dim < info.target_ndim; ++dim) {
        int64_t stride = 1;
        for (int64_t trailing = dim + 1; trailing < info.target_ndim; ++trailing) {
            stride *= target_shape[trailing];
        }
        info.target_strides[dim] = stride;
    }

    const int threads = 256;
    const int blocks = static_cast<int>((input.numel() + threads - 1) / threads);
    if (input.dtype() == DType::ComplexFloat) {
        sum_complex_float_to_shape_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            input.numel(),
            reinterpret_cast<const cuFloatComplex*>(input.data_ptr<std::complex<float>>()),
            reinterpret_cast<cuFloatComplex*>(result.data_ptr<std::complex<float>>()),
            info);
    } else {
        sum_complex_double_to_shape_kernel<<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            input.numel(),
            reinterpret_cast<const cuDoubleComplex*>(input.data_ptr<std::complex<double>>()),
            reinterpret_cast<cuDoubleComplex*>(result.data_ptr<std::complex<double>>()),
            info);
    }
    checkCuda(cudaGetLastError(), "matmul complex sum_to_shape kernel launch");
    return result;
}

Tensor sum_to_shape_cuda(
    const Tensor& input, const std::vector<int64_t>& target_shape) {
    const auto source_shape = static_cast<std::vector<int64_t>>(input.shape());
    if (target_shape.size() > source_shape.size()) {
        TP_THROW(RuntimeError, "matmul backward: target rank exceeds source rank");
    }
    const int64_t leading = static_cast<int64_t>(source_shape.size()) -
                            static_cast<int64_t>(target_shape.size());
    std::vector<int64_t> reduce_dims;
    for (int64_t dim = 0; dim < leading; ++dim) reduce_dims.push_back(dim);
    for (size_t dim = 0; dim < target_shape.size(); ++dim) {
        const int64_t source_dim = source_shape[leading + static_cast<int64_t>(dim)];
        if (target_shape[dim] != 1 && target_shape[dim] != source_dim) {
            TP_THROW(RuntimeError, "matmul backward: incompatible gradient shape");
        }
        if (target_shape[dim] == 1 && source_dim != 1) {
            reduce_dims.push_back(leading + static_cast<int64_t>(dim));
        }
    }
    if (reduce_dims.empty() && source_shape == target_shape) return input;
    if (!reduce_dims.empty() &&
        (input.dtype() == DType::ComplexFloat || input.dtype() == DType::ComplexDouble)) {
        return sum_to_shape_complex_cuda(input, target_shape);
    }
    Tensor reduced = reduce_dims.empty() ? input : input.sum(reduce_dims, true);
    return reduced.reshape(target_shape);
}

struct MatmulBackwardInputsCuda {
    Tensor self_matrix;
    Tensor other_matrix;
    Tensor grad_matrix;
    bool self_vector = false;
    bool other_vector = false;
};

MatmulBackwardInputsCuda normalize_matmul_backward_inputs_cuda(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    MatmulBackwardInputsCuda normalized;
    normalized.self_vector = self.dim() == 1;
    normalized.other_vector = other.dim() == 1;
    normalized.self_matrix = normalized.self_vector ? self.unsqueeze(0) : self;
    normalized.other_matrix = normalized.other_vector ? other.unsqueeze(-1) : other;
    normalized.grad_matrix = grad_output;
    if (normalized.self_vector && normalized.other_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(0).unsqueeze(0);
    } else if (normalized.self_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(-2);
    } else if (normalized.other_vector) {
        normalized.grad_matrix = grad_output.unsqueeze(-1);
    }
    return normalized;
}

Tensor matmul_backward_self_kernel_cuda(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputsCuda normalized =
        normalize_matmul_backward_inputs_cuda(grad_output, self, other);
    Tensor grad = matmul_kernel_cuda(
        normalized.grad_matrix,
        adjoint_last_two_cuda(normalized.other_matrix));
    grad = sum_to_shape_cuda(
        grad, static_cast<std::vector<int64_t>>(normalized.self_matrix.shape()));
    if (normalized.self_vector) grad = grad.squeeze(0);
    if (grad.dtype() != self.dtype()) grad = grad.to(self.dtype());
    return grad;
}

Tensor matmul_backward_other_kernel_cuda(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputsCuda normalized =
        normalize_matmul_backward_inputs_cuda(grad_output, self, other);
    Tensor grad = matmul_kernel_cuda(
        adjoint_last_two_cuda(normalized.self_matrix),
        normalized.grad_matrix);
    grad = sum_to_shape_cuda(
        grad, static_cast<std::vector<int64_t>>(normalized.other_matrix.shape()));
    if (normalized.other_vector) grad = grad.squeeze(-1);
    if (grad.dtype() != other.dtype()) grad = grad.to(other.dtype());
    return grad;
}

Tensor bmm_kernel_cuda(const Tensor& self, const Tensor& batch2) {
    if (self.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    // (B, M, K) @ (B, K, N): batch2's leading dims must match [B, K].
    if (self.size(0) != batch2.size(0) || self.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                 self.size(0), ", ", self.size(2), "] but got: [", batch2.size(0), ", ",
                 batch2.size(1), "].");
    }
    if (self.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", pretty_dtype_name(self.dtype()),
                 " but found ", pretty_dtype_name(batch2.dtype()));
    }
    // Reject unsupported dtypes before touching memory, like torch.
    (void)toCublasType(self.dtype());
    return matmul_batched_2d_cuda(self, batch2, {self.size(0)}, {batch2.size(0)});
}

Tensor baddbmm_kernel_cuda(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                           Scalar beta, Scalar alpha) {
    if (batch1.dim() != 3) TP_THROW(RuntimeError, "batch1 must be a 3D tensor");
    if (batch2.dim() != 3) TP_THROW(RuntimeError, "batch2 must be a 3D tensor");
    if (batch1.size(0) != batch2.size(0) || batch1.size(2) != batch2.size(1)) {
        TP_THROW(RuntimeError, "Expected size for first two dimensions of batch2 tensor to be: [",
                 batch1.size(0), ", ", batch1.size(2), "] but got: [", batch2.size(0), ", ",
                 batch2.size(1), "].");
    }
    if (input.dtype() != batch1.dtype() || batch1.dtype() != batch2.dtype()) {
        TP_THROW(RuntimeError, "Input dtypes must be the same, got: input ",
                 c10_style_dtype_name(input.dtype()), ", batch1: ",
                 c10_style_dtype_name(batch1.dtype()), ", batch2: ",
                 c10_style_dtype_name(batch2.dtype()));
    }

    const int64_t B = batch1.size(0);
    const int64_t M = batch1.size(1);
    const int64_t N = batch2.size(2);
    const double beta_v = beta.toDouble();
    const double alpha_v = alpha.toDouble();

    const std::vector<int64_t> target{B, M, N};
    Tensor result;
    if (beta_v == 0.0) {
        result = Tensor::empty(target, batch1.dtype(), batch1.device());
    } else {
        // Any broadcastable input works in torch, including 0-dim/(N,)/(M,N).
        result = expand_gemm_input_cuda(input, target).contiguous();
        if (beta_v != 1.0) result.mul_(beta);
    }

    Tensor b1 = batch1.is_contiguous() ? batch1 : batch1.contiguous();
    Tensor b2 = batch2.is_contiguous() ? batch2 : batch2.contiguous();
    Tensor result_3d = std::move(result);
    gemm_strided_batched_3d(b1, b2, result_3d, B, M, N, batch1.size(2),
                            alpha_v, beta_v == 0.0 ? 0.0 : 1.0);
    return result_3d;
}

Tensor mv_kernel_cuda(const Tensor& self, const Tensor& vec) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (vec.dim() != 1) TP_THROW(RuntimeError, "vector must be 1D");
    if (self.size(1) != vec.size(0)) {
        TP_THROW(RuntimeError, "size mismatch, got input (", self.size(0), "), mat (",
                 self.size(0), "x", self.size(1), "), vec (", vec.size(0), ")");
    }
    if (self.dtype() != vec.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(vec.dtype()));
    }
    return matmul_batched_2d_cuda(self, vec.unsqueeze(-1), {}, {}).squeeze(-1);
}

Tensor dot_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (self.dim() != 1 || other.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(), "D and ",
                 other.dim(), "D tensors");
    }
    if (self.size(0) != other.size(0)) {
        TP_THROW(RuntimeError, "inconsistent tensor size, expected tensor [", self.size(0),
                 "] and src [", other.size(0),
                 "] to have the same number of elements, but got ", self.size(0), " and ",
                 other.size(0), " elements respective");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(RuntimeError, "dot : expected both vectors to have same dtype, but found ",
                 pretty_dtype_name(self.dtype()), " and ", pretty_dtype_name(other.dtype()));
    }

    const DType dtype = self.dtype();
    // Torch's CUDA dot only implements floating and complex dtypes.
    if (!isFloatingType(dtype) && !isComplexType(dtype)) {
        TP_THROW(NotImplementedError, "\"dot\" not implemented for '",
                 pretty_dtype_name(dtype), "'");
    }

    const int64_t n = self.numel();
    Tensor result = Tensor::empty({}, dtype, self.device());
    if (n == 0) {
        return zero_matmul_output_cuda(result);
    }

    switch (dtype) {
        case DType::Float32:
            CUBLAS_CHECK(cublasSdot(CUDAContext::getCublasHandle(), static_cast<int>(n),
                                    self.data_ptr<float>(), 1, other.data_ptr<float>(), 1,
                                    result.data_ptr<float>()));
            return result;
        case DType::Float64:
            CUBLAS_CHECK(cublasDdot(CUDAContext::getCublasHandle(), static_cast<int>(n),
                                    self.data_ptr<double>(), 1, other.data_ptr<double>(), 1,
                                    result.data_ptr<double>()));
            return result;
        case DType::Float16:
        case DType::BFloat16: {
            // FP32 accumulation contract, native storage for the scalar.
            CUBLAS_CHECK(cublasDotEx(
                CUDAContext::getCublasHandle(), static_cast<int>(n),
                self.data_ptr(), toCublasType(dtype), 1,
                other.data_ptr(), toCublasType(dtype), 1,
                result.data_ptr(), toCublasType(dtype), CUDA_R_32F));
            return result;
        }
        default:
            // Complex dot does not conjugate (that is vdot).
            Tensor product = self * other;
            return product.sum();
    }
}

Tensor inner_kernel_cuda(const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        if (self.dtype() != other.dtype()) {
            TP_THROW(RuntimeError, "expected scalar type ", pretty_dtype_name(self.dtype()),
                     " but found ", pretty_dtype_name(other.dtype()));
        }
        return self.mul(other);
    }
    const int64_t n = self.size(-1);
    if (other.size(-1) != n) {
        TP_THROW(RuntimeError, "inner(): incompatible shapes: ", n, " vs ", other.size(-1));
    }
    if (self.dim() == 1 && other.dim() == 1) {
        return dot_kernel_cuda(self, other);
    }

    Tensor a = self.reshape({-1, n});
    Tensor b = other.reshape({-1, n});
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i + 1 < self.shape().size(); ++i) out_shape.push_back(self.shape()[i]);
    for (size_t i = 0; i + 1 < other.shape().size(); ++i) out_shape.push_back(other.shape()[i]);
    Tensor product = matmul_kernel_cuda(a, transpose_last_two_view_cuda(b));
    return product.reshape(out_shape);
}

Tensor outer_kernel_cuda(const Tensor& self, const Tensor& vec2) {
    if (self.dim() != 1 || vec2.dim() != 1) {
        TP_THROW(RuntimeError, "1D tensors expected, but got ", self.dim(), "D and ",
                 vec2.dim(), "D tensors");
    }
    if (self.dtype() != vec2.dtype()) {
        TP_THROW(RuntimeError, "expected scalar type ", pretty_dtype_name(self.dtype()),
                 " but found ", pretty_dtype_name(vec2.dtype()));
    }
    return self.unsqueeze(1).mul(vec2.unsqueeze(0));
}

Tensor inner_backward_self_kernel_cuda(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        return grad_output * other;
    }
    const int64_t n = std::max<int64_t>(self.size(-1), 1);
    const int64_t prod_a = std::max<int64_t>(self.numel() / n, 1);
    const int64_t prod_b = std::max<int64_t>(other.numel() / n, 1);
    // inner(A, B) == A2 @ B2^T with A2=(prod_a, N), B2=(prod_b, N), so
    // dA2 = grad2 @ B2 -- exactly matmul_backward_self on the flattened pair.
    Tensor grad2 = grad_output.reshape({prod_a, prod_b});
    Tensor other2 = other.reshape({-1, n});
    Tensor da2 = matmul_kernel_cuda(grad2, transpose_last_two_view_cuda(other2));
    Tensor grad = sum_to_shape_cuda(da2, static_cast<std::vector<int64_t>>(self.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(self.shape()));
}

Tensor inner_backward_other_kernel_cuda(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        return grad_output * self;
    }
    const int64_t n = std::max<int64_t>(self.size(-1), 1);
    const int64_t prod_a = std::max<int64_t>(self.numel() / n, 1);
    const int64_t prod_b = std::max<int64_t>(other.numel() / n, 1);
    // dB2^T = A2^T @ grad2 -- exactly matmul_backward_other on the flat pair.
    Tensor grad2 = grad_output.reshape({prod_a, prod_b});
    Tensor self2 = self.reshape({-1, n});
    Tensor db2t = matmul_kernel_cuda(transpose_last_two_view_cuda(self2), grad2);
    Tensor db2 = transpose_last_two_view_cuda(db2t);
    Tensor grad = sum_to_shape_cuda(db2, static_cast<std::vector<int64_t>>(other.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(other.shape()));
}

TENSORPLAY_LIBRARY_IMPL(CUDA, LinearAlgebraKernels) {
    m.impl("mm", mm_kernel_cuda);
    m.impl("matmul", matmul_kernel_cuda);
    m.impl("matmul.out", matmul_out_kernel_cuda);
    m.impl("matmul_backward_self", matmul_backward_self_kernel_cuda);
    m.impl("matmul_backward_other", matmul_backward_other_kernel_cuda);
    m.impl("addmm", addmm_kernel_cuda);
    m.impl("bmm", bmm_kernel_cuda);
    m.impl("baddbmm", baddbmm_kernel_cuda);
    m.impl("mv", mv_kernel_cuda);
    m.impl("dot", dot_kernel_cuda);
    m.impl("inner", inner_kernel_cuda);
    m.impl("inner_backward_self", inner_backward_self_kernel_cuda);
    m.impl("inner_backward_other", inner_backward_other_kernel_cuda);
    m.impl("outer", outer_kernel_cuda);
}

} // namespace cuda
} // namespace tensorplay
