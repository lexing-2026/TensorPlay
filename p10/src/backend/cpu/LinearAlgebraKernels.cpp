#include "Tensor.h"
#include "TypePromotion.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Parallel.h"
#include "Utils.h"
#include "OneDNNContext.h"
#include "GradMode.h"
#include "LinearAlgebraNames.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <complex>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>


#ifdef USE_MKL
#include <mkl.h>
#elif defined(USE_BLAS)
#include <cblas.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tensorplay {
namespace cpu {
using namespace tensorplay::parallel;

namespace {

void check_cpu_matmul_dtype(DType dtype) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
            // Torch routes CPU matmul through addmm_impl_cpu_, whose dtype
            // rejection reads "addmm_impl_cpu_" not implemented for 'Bool'.
            TP_THROW(NotImplementedError, "\"addmm_impl_cpu_\" not implemented for '",
                     pretty_dtype_name(dtype), "'");
        case DType::ComplexHalf:
        case DType::BComplex32:
            TP_THROW(NotImplementedError, "\"addmm_impl_cpu_\" not implemented for '",
                     pretty_dtype_name(dtype), "'");
        default:
            return;
    }
}

// Naive matrix multiplication implementation (Optimized Loop Order M-K-N)
void gemm_naive(int64_t M, int64_t N, int64_t K, float alpha, const float* A, int64_t lda, const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
    // Scale C by beta first
    if (beta == 0.0f) {
        parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            std::memset(C + m * ldc, 0, N * sizeof(float));
        }
        });
    } else if (beta != 1.0f) {
        parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] *= beta;
            }
        }
        });
    }

    // Accumulate alpha * A * B
    // Loop order M-K-N optimizes cache access for RowMajor matrices B and C
    parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
    for (int64_t m = begin; m < end; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            float a_val = alpha * A[m * lda + k];
            // Vectorization friendly inner loop
            for (int64_t n = 0; n < N; ++n) {
                C[m * ldc + n] += a_val * B[k * ldb + n];
            }
        }
    }
    });
}

// Generic strided GEMM used by matmul for dtypes that do not have a BLAS or
// oneDNN fast path.  The inputs are deliberately addressed through strides so
// batched matmul can consume transposed and broadcast-selected matrix views
// without flattening an expanded (zero-stride) tensor.
template <typename T>
struct MatmulAccumType {
    using type = T;
};

// Integral matmul returns the input dtype.  Accumulate in unsigned arithmetic
// wide enough to make the two's-complement wrap explicit instead of relying on
// signed-overflow behavior; the low bits are the same values Torch stores.
template <> struct MatmulAccumType<int8_t> { using type = uint32_t; };
template <> struct MatmulAccumType<int16_t> { using type = uint32_t; };
template <> struct MatmulAccumType<int32_t> { using type = uint64_t; };
template <> struct MatmulAccumType<int64_t> { using type = unsigned __int128; };
template <> struct MatmulAccumType<uint8_t> { using type = uint32_t; };
template <> struct MatmulAccumType<uint16_t> { using type = uint32_t; };
template <> struct MatmulAccumType<uint32_t> { using type = uint64_t; };
template <> struct MatmulAccumType<uint64_t> { using type = unsigned __int128; };
template <> struct MatmulAccumType<Half> { using type = float; };
template <> struct MatmulAccumType<BFloat16> { using type = float; };
template <> struct MatmulAccumType<std::complex<Half>> { using type = std::complex<float>; };
template <> struct MatmulAccumType<std::complex<BFloat16>> { using type = std::complex<float>; };

template <typename T>
typename MatmulAccumType<T>::type matmul_to_accum(const T& value) {
    return static_cast<typename MatmulAccumType<T>::type>(value);
}

inline std::complex<float> matmul_to_accum(const std::complex<Half>& value) {
    return {static_cast<float>(value.real()), static_cast<float>(value.imag())};
}

inline std::complex<float> matmul_to_accum(const std::complex<BFloat16>& value) {
    return {static_cast<float>(value.real()), static_cast<float>(value.imag())};
}

template <typename T>
void gemm_strided(
    int64_t M, int64_t N, int64_t K,
    const T* A, int64_t a_stride0, int64_t a_stride1,
    const T* B, int64_t b_stride0, int64_t b_stride1,
    T* C, int64_t c_stride0, int64_t c_stride1) {
    using AccT = typename MatmulAccumType<T>::type;
    parallel_for(0, M, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                AccT acc{};
                for (int64_t k = 0; k < K; ++k) {
                    acc += matmul_to_accum(A[m * a_stride0 + k * a_stride1]) *
                           matmul_to_accum(B[k * b_stride0 + n * b_stride1]);
                }
                C[m * c_stride0 + n * c_stride1] = static_cast<T>(acc);
            }
        }
    });
}

void gemm_strided_dispatch(
    const Tensor& self, const Tensor& other, Tensor& result,
    int64_t M, int64_t N, int64_t K) {
    if (self.dtype() == DType::Bool) {
        // PyTorch's CPU addmm/matmul backend does not implement Bool GEMM.
        TP_THROW(NotImplementedError, "matmul: Bool is not supported on CPU");
    }

#define MATMUL_CASE(ctype, name) \
    case DType::name: \
        gemm_strided<ctype>(M, N, K, \
            self.data_ptr<ctype>(), self.stride(0), self.stride(1), \
            other.data_ptr<ctype>(), other.stride(0), other.stride(1), \
            result.data_ptr<ctype>(), result.stride(0), result.stride(1)); \
        return;

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(MATMUL_CASE)
        case DType::ComplexHalf:
            gemm_strided<std::complex<Half>>(
                M, N, K, self.data_ptr<std::complex<Half>>(), self.stride(0), self.stride(1),
                other.data_ptr<std::complex<Half>>(), other.stride(0), other.stride(1),
                result.data_ptr<std::complex<Half>>(), result.stride(0), result.stride(1));
            return;
        case DType::ComplexFloat:
            gemm_strided<std::complex<float>>(
                M, N, K, self.data_ptr<std::complex<float>>(), self.stride(0), self.stride(1),
                other.data_ptr<std::complex<float>>(), other.stride(0), other.stride(1),
                result.data_ptr<std::complex<float>>(), result.stride(0), result.stride(1));
            return;
        case DType::ComplexDouble:
            gemm_strided<std::complex<double>>(
                M, N, K, self.data_ptr<std::complex<double>>(), self.stride(0), self.stride(1),
                other.data_ptr<std::complex<double>>(), other.stride(0), other.stride(1),
                result.data_ptr<std::complex<double>>(), result.stride(0), result.stride(1));
            return;
        case DType::BComplex32:
            gemm_strided<std::complex<BFloat16>>(
                M, N, K, self.data_ptr<std::complex<BFloat16>>(), self.stride(0), self.stride(1),
                other.data_ptr<std::complex<BFloat16>>(), other.stride(0), other.stride(1),
                result.data_ptr<std::complex<BFloat16>>(), result.stride(0), result.stride(1));
            return;
        default:
            TP_THROW(NotImplementedError, "matmul: unsupported dtype on CPU");
    }
#undef MATMUL_CASE
}

#ifdef USE_ONEDNN
using namespace dnnl;

bool mm_onednn(const Tensor& self, const Tensor& mat2, Tensor& result) {
    if (!OneDNNContext::is_enabled()) return false;
    if (self.dtype() != DType::Float32 || mat2.dtype() != DType::Float32) return false;

    // Dimensions
    int64_t M = self.size(0);
    int64_t K = self.size(1);
    int64_t N = mat2.size(1);

    try {
        auto& engine = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();

        // Memory descriptors with explicit strides
        memory::dims src_dims = {M, K};
        memory::dims src_strides = {self.stride(0), self.stride(1)};
        auto src_md = memory::desc(src_dims, memory::data_type::f32, src_strides);

        memory::dims weights_dims = {K, N};
        memory::dims weights_strides = {mat2.stride(0), mat2.stride(1)};
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, weights_strides);

        memory::dims dst_dims = {M, N};
        memory::dims dst_strides = {result.stride(0), result.stride(1)};
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_strides);

        // Create memories sharing data pointers
        auto src_mem = memory(src_md, engine, self.data_ptr<float>());
        auto weights_mem = memory(weights_md, engine, mat2.data_ptr<float>());
        auto dst_mem = memory(dst_md, engine, result.data_ptr<float>());

        // Primitive descriptor
        auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);

        // Primitive
        auto matmul_prim = matmul(matmul_pd);

        // Execute
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem}
        });

        stream.wait();
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
}

bool matmul_onednn(const Tensor& src, const Tensor& weights, Tensor& dst) {
    if (!OneDNNContext::is_enabled()) return false;
    if (src.dtype() != DType::Float32 || weights.dtype() != DType::Float32) return false;

    try {
        auto& engine = OneDNNContext::get_engine();
        auto& stream = OneDNNContext::get_stream();

        // Convert shapes and strides to memory::dims
        memory::dims src_dims = static_cast<std::vector<int64_t>>(src.shape());
        memory::dims src_strides = static_cast<std::vector<int64_t>>(src.strides());
        auto src_md = memory::desc(src_dims, memory::data_type::f32, src_strides);

        memory::dims weights_dims = static_cast<std::vector<int64_t>>(weights.shape());
        memory::dims weights_strides = static_cast<std::vector<int64_t>>(weights.strides());
        auto weights_md = memory::desc(weights_dims, memory::data_type::f32, weights_strides);

        memory::dims dst_dims = static_cast<std::vector<int64_t>>(dst.shape());
        memory::dims dst_strides = static_cast<std::vector<int64_t>>(dst.strides());
        auto dst_md = memory::desc(dst_dims, memory::data_type::f32, dst_strides);

        // Create memories sharing data pointers
        auto src_mem = memory(src_md, engine, src.data_ptr<float>());
        auto weights_mem = memory(weights_md, engine, weights.data_ptr<float>());
        auto dst_mem = memory(dst_md, engine, dst.data_ptr<float>());

        // Primitive descriptor
        auto matmul_pd = matmul::primitive_desc(engine, src_md, weights_md, dst_md);

        // Primitive
        auto matmul_prim = matmul(matmul_pd);

        // Execute
        matmul_prim.execute(stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_WEIGHTS, weights_mem},
            {DNNL_ARG_DST, dst_mem}
        });

        stream.wait();
        return true;
    } catch (dnnl::error& e) {
        return false;
    }
}
#endif

} // anonymous namespace

Tensor mm_kernel(const Tensor& self, const Tensor& mat2) {
    if (self.dim() != 2) TP_THROW(RuntimeError, "self must be a matrix");
    if (mat2.dim() != 2) TP_THROW(RuntimeError, "mat2 must be a matrix");
    if (self.size(1) != mat2.size(0)) {
        // Torch wording, e.g. "mat1 and mat2 shapes cannot be multiplied (2x3 and 5x4)".
        TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", self.size(0), "x", self.size(1),
                 " and ", mat2.size(0), "x", mat2.size(1), ")");
    }

    // torch.mm/matmul require the two matrix operands to have the same
    // dtype.  This is intentionally stricter than elementwise promotion.
    if (self.dtype() != mat2.dtype()) {
        TP_THROW(RuntimeError, "expected m1 and m2 to have the same dtype, but got: ",
                 c10_style_dtype_name(self.dtype()), " != ", c10_style_dtype_name(mat2.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());
    const DType result_dtype = self.dtype();
    const Tensor& self_p = self;
    const Tensor& mat2_p = mat2;
    
    int64_t M = self_p.size(0);
    int64_t K = self_p.size(1);
    int64_t N = mat2_p.size(1);
    
    Tensor result = Tensor::empty({M, N}, result_dtype, self.device());
    if (K == 0) {
        // oneDNN and some BLAS implementations leave C untouched for a
        // zero-inner-dimension GEMM, while torch.matmul defines the result as
        // beta*C + 0 and beta is zero for mm.
        return result.fill_(Scalar(0));
    }
    
    #ifdef USE_ONEDNN
    if (mm_onednn(self_p, mat2_p, result)) {
        return result;
    }
    #endif

    if (self_p.dtype() == DType::Float32 && mat2_p.dtype() == DType::Float32) {
        
        bool transA = false;
        bool transB = false;
        int64_t lda = 0;
        int64_t ldb = 0;
        
        // Helper to check for transposed layout (stride(0)=1, stride(1)=size(0))
        // This corresponds to a Column-Major layout of a matrix of size (size(0), size(1))
        // or a Transposed view of a Row-Major matrix.
        auto is_transposed = [](const Tensor& t) {
            return t.stride(0) == 1 && t.stride(1) == t.size(0);
        };
        
        Tensor a_input = self_p;
        if (self_p.is_contiguous()) {
            lda = K;
        } else if (is_transposed(self_p)) {
            transA = true;
            lda = M; 
        } else {
            a_input = self_p.clone();
            lda = K;
        }
        
        Tensor b_input = mat2_p;
        if (mat2_p.is_contiguous()) {
            ldb = N;
        } else if (is_transposed(mat2_p)) {
            transB = true;
            ldb = K;
        } else {
            b_input = mat2_p.clone();
            ldb = N;
        }
        
        const float* A = a_input.data_ptr<float>();
        const float* B = b_input.data_ptr<float>();
        float* C = result.data_ptr<float>();
        
        #if defined(USE_MKL) || defined(USE_BLAS)
            CBLAS_TRANSPOSE TransA = transA ? CblasTrans : CblasNoTrans;
            CBLAS_TRANSPOSE TransB = transB ? CblasTrans : CblasNoTrans;
            
            #ifdef USE_MKL
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        M, N, K, 1.0f, A, lda, B, ldb, 0.0f, C, N);
            #else
            cblas_sgemm(CblasRowMajor, TransA, TransB, 
                        (int)M, (int)N, (int)K, 1.0f, A, (int)lda, B, (int)ldb, 0.0f, C, (int)N);
            #endif
            
        #else
            // Fallback to naive (no transpose support in naive yet, force clone)
            if (transA || transB) {
                 Tensor a_contig = self_p.is_contiguous() ? self_p : self_p.clone();
                 Tensor b_contig = mat2_p.is_contiguous() ? mat2_p : mat2_p.clone();
                 gemm_naive(M, N, K, 1.0f, a_contig.data_ptr<float>(), K, b_contig.data_ptr<float>(), N, 0.0f, C, N);
            } else {
                 gemm_naive(M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
            }
        #endif
        
    } else {
        // Keep the generic path stride-aware so mm/matmul also work for
        // float64, reduced-float, integral and complex tensors where the
        // optimized float32 kernels are unavailable.
        gemm_strided_dispatch(self_p, mat2_p, result, M, N, K);
    }
    
    return result;
}

// Torch broadcasts `input` against the GEMM output shape right-aligned and
// reports mismatches through its expand wording.  Returns a (possibly
// zero-stride) view; callers materialize with clone() before mutating.
Tensor expand_gemm_input(const Tensor& input, const std::vector<int64_t>& target) {
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

    // Validate right-to-left so the reported dimension matches torch's.
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

Tensor addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2, Scalar beta, Scalar alpha) {
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

    // out = beta * input + alpha * (mat1 @ mat2)
    Tensor result;
    if (beta_v == 0.0) {
        result = Tensor::empty({M, N}, mat1.dtype(), mat1.device());
    } else {
        // Any broadcastable input works in torch, including 0-dim/(M,1)/(1,N).
        result = expand_gemm_input(input, {M, N}).clone();
        if (beta_v != 1.0) result.mul_(beta);
    }

    // alpha * (self @ other) via mm_kernel, then add
    Tensor mm = mm_kernel(mat1, mat2);
    if (alpha_v != 1.0) {
        mm = mm.mul(alpha_v);
    }
    result = result.add(mm, 1.0);
    return result;
}

std::vector<int64_t> decode_batch_index(
    int64_t linear_index, const std::vector<int64_t>& batch_shape) {
    std::vector<int64_t> index(batch_shape.size(), 0);
    for (int64_t dim = static_cast<int64_t>(batch_shape.size()) - 1; dim >= 0; --dim) {
        const int64_t size = batch_shape[dim];
        index[dim] = size == 0 ? 0 : linear_index % size;
        if (size != 0) linear_index /= size;
    }
    return index;
}

// Select one matrix from a possibly broadcast batch.  Dimensions are removed
// from high to low so the original dimension numbers remain valid.
Tensor select_batch_matrix(
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

Tensor select_output_matrix(
    const Tensor& output,
    const std::vector<int64_t>& output_index) {
    Tensor matrix = output;
    for (int64_t dim = static_cast<int64_t>(output_index.size()) - 1; dim >= 0; --dim) {
        matrix = matrix.select(dim, output_index[dim]);
    }
    return matrix;
}

Tensor matmul_batched_2d(
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

    if (K == 0) {
        return result.fill_(Scalar(0));
    }

    // Preserve the oneDNN N-D fast path when neither operand needs an
    // implicit broadcast/padding. Expanded views can have zero strides,
    // which oneDNN rightfully rejects.
#ifdef USE_ONEDNN
    if (self_batch_shape == batch_shape && other_batch_shape == batch_shape &&
        self.dim() == static_cast<int64_t>(batch_shape.size()) + 2 &&
        other.dim() == static_cast<int64_t>(batch_shape.size()) + 2 &&
        matmul_onednn(self, other, result)) {
        return result;
    }
#endif

    for (int64_t linear = 0; linear < batch_size; ++linear) {
        const std::vector<int64_t> output_index = decode_batch_index(linear, batch_shape);
        Tensor self_matrix = select_batch_matrix(self, self_batch_shape, batch_shape, output_index);
        Tensor other_matrix = select_batch_matrix(other, other_batch_shape, batch_shape, output_index);
        Tensor matrix_result = mm_kernel(self_matrix, other_matrix);
        select_output_matrix(result, output_index).copy_(matrix_result);
    }
    return result;
}

Tensor matmul_kernel(const Tensor& self, const Tensor& other) {
    const int64_t dim1 = self.dim();
    const int64_t dim2 = other.dim();
    if (self.dim() < 1 || other.dim() < 1) {
        TP_THROW(RuntimeError, "matmul(): input operands must be at least 1D");
    }

    // Shape contract first, using torch's exact wording.  Torch folds
    // vector/batch dimensions into matrix rows before reporting mismatches,
    // so the reported shapes depend on operand ranks:
    //   vec @ mat      -> "mat1 and mat2 shapes cannot be multiplied (1xN and KxM)"
    //   mat @ vec      -> "size mismatch, got input (M), mat (MxK), vec (V)"
    //   batched @ mat  -> batch*M folded into rows of mat1
    //   * @ batched    -> "Expected size for first two dimensions of batch2 tensor ..."
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
        // (batch..., M, K) @ (K,) folds batch into mat1's rows like torch.
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
                TP_THROW(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (", folded_m, "x", k,
                         " and ", other_k, "x", other_shape.back(), ")");
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
    check_cpu_matmul_dtype(self.dtype());
    const Tensor& self_p = self;
    const Tensor& other_p = other;

    if (dim1 == 1 && dim2 == 1) {
        if (self_p.size(0) != other_p.size(0)) {
            TP_THROW(RuntimeError, "matmul: size mismatch, got ", self_p.size(0), " and ", other_p.size(0));
        }
        Tensor result = matmul_batched_2d(
            self_p.unsqueeze(0), other_p.unsqueeze(1), {}, {});
        return result.squeeze(0).squeeze(0);
    }

    if (dim1 == 1) {
        std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
        Tensor result = matmul_batched_2d(
            self_p.unsqueeze(0), other_p, {}, other_batch_shape);
        return result.squeeze(-2);
    }

    if (dim2 == 1) {
        std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
        Tensor result = matmul_batched_2d(
            self_p, other_p.unsqueeze(-1), self_batch_shape, {});
        return result.squeeze(-1);
    }

    std::vector<int64_t> self_batch_shape(self_shape.begin(), self_shape.end() - 2);
    std::vector<int64_t> other_batch_shape(other_shape.begin(), other_shape.end() - 2);
    return matmul_batched_2d(self_p, other_p, self_batch_shape, other_batch_shape);
}

Tensor& matmul_out_kernel(const Tensor& self, const Tensor& other, Tensor& out) {
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
    Tensor result = matmul_kernel(self, other);
    if (out.shape() == result.shape()) {
        out.copy_(result);
    } else {
        // Out variants resize a write-only destination to the inferred shape.
        // Replacing metadata also gives the resized tensor fresh contiguous
        // storage, matching TensorIterator's write-only resize contract.
        out.unsafeGetTensorImpl()->copy_metadata_from(*result.unsafeGetTensorImpl());
    }
    return out;
}

namespace {

Tensor transpose_last_two_view(const Tensor& input) {
    if (input.dim() < 2) {
        TP_THROW(RuntimeError, "matmul backward: expected a matrix operand");
    }
    std::vector<int64_t> sizes = static_cast<std::vector<int64_t>>(input.shape());
    std::vector<int64_t> strides = input.strides();
    std::swap(sizes[sizes.size() - 2], sizes[sizes.size() - 1]);
    std::swap(strides[strides.size() - 2], strides[strides.size() - 1]);
    return input.as_strided(sizes, strides);
}

template <typename Component>
Tensor conjugate_contiguous_cpu(const Tensor& input) {
    Tensor result = Tensor::empty(
        static_cast<std::vector<int64_t>>(input.shape()), input.dtype(), input.device());
    const auto* src = input.data_ptr<std::complex<Component>>();
    auto* dst = result.data_ptr<std::complex<Component>>();
    parallel_for(0, input.numel(), GRAIN_SIZE, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) dst[i] = std::conj(src[i]);
    });
    return result;
}

Tensor adjoint_last_two_cpu(const Tensor& input) {
    Tensor transposed = transpose_last_two_view(input);
    if (input.dtype() == DType::ComplexFloat) {
        return conjugate_contiguous_cpu<float>(transposed.contiguous());
    }
    if (input.dtype() == DType::ComplexDouble) {
        return conjugate_contiguous_cpu<double>(transposed.contiguous());
    }
    return transposed;
}

template <typename T>
void sum_to_shape_recursive(
    const T* src, const std::vector<int64_t>& src_shape,
    const std::vector<int64_t>& src_strides,
    T* dst, const std::vector<int64_t>& dst_shape,
    const std::vector<int64_t>& dst_strides,
    int64_t dim, int64_t src_offset, int64_t dst_offset) {
    if (dim == static_cast<int64_t>(src_shape.size())) {
        dst[dst_offset] = static_cast<T>(
            matmul_to_accum(dst[dst_offset]) + matmul_to_accum(src[src_offset]));
        return;
    }

    const int64_t leading = static_cast<int64_t>(src_shape.size()) -
                            static_cast<int64_t>(dst_shape.size());
    const int64_t dst_dim = dim - leading;
    const bool reduce = dst_dim < 0 || dst_shape[dst_dim] == 1;
    const int64_t dst_stride = (dst_dim < 0 || reduce) ? 0 : dst_strides[dst_dim];
    for (int64_t i = 0; i < src_shape[dim]; ++i) {
        sum_to_shape_recursive(
            src, src_shape, src_strides, dst, dst_shape, dst_strides,
            dim + 1, src_offset + i * src_strides[dim],
            dst_offset + (reduce ? 0 : i * dst_stride));
    }
}

Tensor sum_to_shape_cpu(const Tensor& input, const std::vector<int64_t>& target_shape) {
    const auto source_shape = static_cast<std::vector<int64_t>>(input.shape());
    if (target_shape.size() > source_shape.size()) {
        TP_THROW(RuntimeError, "matmul backward: target rank exceeds source rank");
    }
    const int64_t leading = static_cast<int64_t>(source_shape.size()) -
                            static_cast<int64_t>(target_shape.size());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        const int64_t source_dim = source_shape[leading + static_cast<int64_t>(i)];
        if (target_shape[i] != 1 && target_shape[i] != source_dim) {
            TP_THROW(RuntimeError, "matmul backward: incompatible gradient shape");
        }
    }

    if (source_shape == target_shape) return input;
    Tensor result = Tensor::zeros(target_shape, input.dtype(), input.device());

#define SUM_TO_CASE(ctype, name) \
    case DType::name: \
        sum_to_shape_recursive<ctype>( \
            input.data_ptr<ctype>(), source_shape, input.strides(), \
            result.data_ptr<ctype>(), target_shape, result.strides(), 0, 0, 0); \
        return result;

    switch (input.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(SUM_TO_CASE)
        case DType::ComplexHalf:
            sum_to_shape_recursive<std::complex<Half>>(
                input.data_ptr<std::complex<Half>>(), source_shape, input.strides(),
                result.data_ptr<std::complex<Half>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::ComplexFloat:
            sum_to_shape_recursive<std::complex<float>>(
                input.data_ptr<std::complex<float>>(), source_shape, input.strides(),
                result.data_ptr<std::complex<float>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::ComplexDouble:
            sum_to_shape_recursive<std::complex<double>>(
                input.data_ptr<std::complex<double>>(), source_shape, input.strides(),
                result.data_ptr<std::complex<double>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        case DType::BComplex32:
            sum_to_shape_recursive<std::complex<BFloat16>>(
                input.data_ptr<std::complex<BFloat16>>(), source_shape, input.strides(),
                result.data_ptr<std::complex<BFloat16>>(), target_shape, result.strides(), 0, 0, 0);
            return result;
        default:
            TP_THROW(NotImplementedError, "matmul backward: unsupported dtype");
    }
#undef SUM_TO_CASE
}

struct MatmulBackwardInputs {
    Tensor self_matrix;
    Tensor other_matrix;
    Tensor grad_matrix;
    std::vector<int64_t> self_shape;
    std::vector<int64_t> other_shape;
    bool self_vector = false;
    bool other_vector = false;
};

MatmulBackwardInputs normalize_matmul_backward_inputs(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    MatmulBackwardInputs normalized;
    normalized.self_vector = self.dim() == 1;
    normalized.other_vector = other.dim() == 1;
    normalized.self_shape = static_cast<std::vector<int64_t>>(self.shape());
    normalized.other_shape = static_cast<std::vector<int64_t>>(other.shape());
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

} // namespace

Tensor matmul_backward_self_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputs normalized =
        normalize_matmul_backward_inputs(grad_output, self, other);
    Tensor grad = matmul_kernel(
        normalized.grad_matrix,
        adjoint_last_two_cpu(normalized.other_matrix));
    std::vector<int64_t> target_shape = static_cast<std::vector<int64_t>>(
        normalized.self_matrix.shape());
    grad = sum_to_shape_cpu(grad, target_shape);
    if (normalized.self_vector) grad = grad.squeeze(0);
    if (grad.dtype() != self.dtype()) grad = grad.to(self.dtype());
    return grad;
}

Tensor matmul_backward_other_kernel(
    const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    const MatmulBackwardInputs normalized =
        normalize_matmul_backward_inputs(grad_output, self, other);
    Tensor grad = matmul_kernel(
        adjoint_last_two_cpu(normalized.self_matrix),
        normalized.grad_matrix);
    std::vector<int64_t> target_shape = static_cast<std::vector<int64_t>>(
        normalized.other_matrix.shape());
    grad = sum_to_shape_cpu(grad, target_shape);
    if (normalized.other_vector) grad = grad.squeeze(-1);
    if (grad.dtype() != other.dtype()) grad = grad.to(other.dtype());
    return grad;
}

Tensor bmm_kernel(const Tensor& self, const Tensor& batch2) {
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
    check_cpu_matmul_dtype(self.dtype());
    return matmul_batched_2d(self, batch2, {self.size(0)}, {batch2.size(0)});
}

Tensor baddbmm_kernel(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                      Scalar beta, Scalar alpha) {
    // Torch validates the GEMM shapes through the broadcast of the result
    // against `input`, so a wrong K shows up as an expand error.
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
    check_cpu_matmul_dtype(batch1.dtype());

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
        result = expand_gemm_input(input, target).clone();
        if (beta_v != 1.0) result.mul_(beta);
    }

    Tensor product = bmm_kernel(batch1, batch2);
    if (alpha_v != 1.0) product = product.mul(alpha_v);
    return result.add(product, 1.0);
}

Tensor mv_kernel(const Tensor& self, const Tensor& vec) {
    // Torch routes mv through addmv; mirror its meta checks verbatim
    // (aten/src/ATen/native/Blas.cpp ADDMV_META).
    if (self.dim() != 2 || vec.dim() != 1) {
        TP_THROW(RuntimeError, "vector + matrix @ vector expected, got ", 1, ", ",
                 self.dim(), ", ", vec.dim());
    }
    if (self.size(1) != vec.size(0)) {
        TP_THROW(RuntimeError, "size mismatch, got input (", self.size(0), "), mat (",
                 self.size(0), "x", self.size(1), "), vec (", vec.size(0), ")");
    }
    if (self.dtype() != vec.dtype()) {
        // The leading value is addmv's accumulator slot; via mv it echoes vec.
        TP_THROW(RuntimeError, "addmv input tensors must have the same dtype, but got ",
                 pretty_dtype_name(vec.dtype()), ", ", pretty_dtype_name(self.dtype()), ", and ",
                 pretty_dtype_name(vec.dtype()));
    }
    check_cpu_matmul_dtype(self.dtype());
    return matmul_batched_2d(self, vec.unsqueeze(-1), {}, {}).squeeze(-1);
}

Tensor dot_kernel(const Tensor& self, const Tensor& other) {
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

    const int64_t n = self.numel();
    Tensor result = Tensor::empty({}, self.dtype(), self.device());
    const auto accumulate = [&](auto&& body) {
        using Acc = decltype(body(int64_t{0}));
        Acc total{};
        parallel_for(0, n, GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            Acc part{};
            for (int64_t i = begin; i < end; ++i) part += body(i);
            // Serial combine keeps the reduction deterministic.
            static std::mutex m;
            std::lock_guard<std::mutex> lock(m);
            total += part;
        });
        return total;
    };

#define DOT_CASE(ctype, name) \
    case DType::name: { \
        if constexpr (std::is_same_v<ctype, bool>) { \
            TP_THROW(NotImplementedError, "\"dot\" not implemented for 'Bool'"); \
        } else { \
            const ctype* a = self.data_ptr<ctype>(); \
            const ctype* b = other.data_ptr<ctype>(); \
            ctype out = static_cast<ctype>(accumulate([&](int64_t i) { \
                return matmul_to_accum(a[i]) * matmul_to_accum(b[i]); })); \
            result.data_ptr<ctype>()[0] = out; \
            return result; \
        } \
    }
    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(DOT_CASE)
        case DType::ComplexHalf:
        case DType::ComplexFloat:
        case DType::ComplexDouble:
        case DType::BComplex32:
            // Complex dot does not conjugate (that is vdot).
            switch (self.dtype()) {
                case DType::ComplexHalf: {
                    using c = std::complex<Half>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    std::complex<float> out{};
                    for (int64_t i = 0; i < n; ++i)
                        out += matmul_to_accum(a[i]) * matmul_to_accum(b[i]);
                    result.data_ptr<c>()[0] = static_cast<c>(out);
                    return result;
                }
                case DType::ComplexFloat: {
                    using c = std::complex<float>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    c out{};
                    for (int64_t i = 0; i < n; ++i) out += a[i] * b[i];
                    result.data_ptr<c>()[0] = out;
                    return result;
                }
                case DType::ComplexDouble: {
                    using c = std::complex<double>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    c out{};
                    for (int64_t i = 0; i < n; ++i) out += a[i] * b[i];
                    result.data_ptr<c>()[0] = out;
                    return result;
                }
                default: {
                    using c = std::complex<BFloat16>;
                    const c* a = self.data_ptr<c>();
                    const c* b = other.data_ptr<c>();
                    std::complex<float> out{};
                    for (int64_t i = 0; i < n; ++i)
                        out += matmul_to_accum(a[i]) * matmul_to_accum(b[i]);
                    result.data_ptr<c>()[0] = static_cast<c>(out);
                    return result;
                }
            }
        default:
            TP_THROW(NotImplementedError, "\"dot\" not implemented for '",
                     pretty_dtype_name(self.dtype()), "'");
    }
#undef DOT_CASE
}

Tensor inner_kernel(const Tensor& self, const Tensor& other) {
    // Torch: scalar operands go through plain multiplication (with
    // promotion); otherwise this is tensordot over the last dimension.
    if (self.dim() == 0 || other.dim() == 0) {
        return self * other;
    }
    const int64_t n = self.size(-1);
    if (other.size(-1) != n) {
        TP_THROW(RuntimeError, "inner() the last dimension must match on both input tensors but got shapes ",
                 Size(static_cast<std::vector<int64_t>>(self.shape())).toString(), " and ",
                 Size(static_cast<std::vector<int64_t>>(other.shape())).toString());
    }
    if (self.dim() == 1 && other.dim() == 1) {
        return dot_kernel(self, other);
    }

    // Contract the last dimension: flatten all leading dims to rows and run
    // one (batched) GEMM against the transposed partner, then restore shape.
    Tensor a = self.reshape({-1, n});
    Tensor b = other.reshape({-1, n});
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i + 1 < self.shape().size(); ++i) out_shape.push_back(self.shape()[i]);
    for (size_t i = 0; i + 1 < other.shape().size(); ++i) out_shape.push_back(other.shape()[i]);
    Tensor product = matmul_kernel(a, transpose_last_two_view(b));
    return product.reshape(out_shape);
}

Tensor outer_kernel(const Tensor& self, const Tensor& vec2) {
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

Tensor inner_backward_self_kernel(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
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
    Tensor da2 = matmul_kernel(grad2, transpose_last_two_view(other2));
    Tensor grad = sum_to_shape_cpu(da2, static_cast<std::vector<int64_t>>(self.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(self.shape()));
}

Tensor inner_backward_other_kernel(const Tensor& grad_output, const Tensor& self, const Tensor& other) {
    if (self.dim() == 0 || other.dim() == 0) {
        return grad_output * self;
    }
    const int64_t n = std::max<int64_t>(self.size(-1), 1);
    const int64_t prod_a = std::max<int64_t>(self.numel() / n, 1);
    const int64_t prod_b = std::max<int64_t>(other.numel() / n, 1);
    // dB2^T = A2^T @ grad2 -- exactly matmul_backward_other on the flat pair.
    Tensor grad2 = grad_output.reshape({prod_a, prod_b});
    Tensor self2 = self.reshape({-1, n});
    Tensor db2t = matmul_kernel(transpose_last_two_view(self2), grad2);
    Tensor db2 = transpose_last_two_view(db2t);
    Tensor grad = sum_to_shape_cpu(db2, static_cast<std::vector<int64_t>>(other.shape()));
    return grad.reshape(static_cast<std::vector<int64_t>>(other.shape()));
}

TENSORPLAY_LIBRARY_IMPL(CPU, LinearAlgebraKernels) {
    m.impl("mm", mm_kernel);
    m.impl("matmul", matmul_kernel);
    m.impl("matmul.out", matmul_out_kernel);
    m.impl("matmul_backward_self", matmul_backward_self_kernel);
    m.impl("matmul_backward_other", matmul_backward_other_kernel);
    m.impl("addmm", addmm_kernel);
    m.impl("bmm", bmm_kernel);
    m.impl("baddbmm", baddbmm_kernel);
    m.impl("mv", mv_kernel);
    m.impl("dot", dot_kernel);
    m.impl("inner", inner_kernel);
    m.impl("inner_backward_self", inner_backward_self_kernel);
    m.impl("inner_backward_other", inner_backward_other_kernel);
    m.impl("outer", outer_kernel);
}

} // namespace cpu
} // namespace tensorplay
