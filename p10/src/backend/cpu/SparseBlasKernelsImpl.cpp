// ISA-tier copies of the sampled SpMM kernel (see TP_CPU_KERNEL_SRCS in
// p10/CMakeLists.txt).  Each capability tier lands in its own CPU_CAPABILITY
// inline namespace and registers its slot on sampled_addmm_stub; the base
// entry point in SparseBlasKernels.cpp calls the stub.

#include "cpu/SparseBlasKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <cstdint>

namespace tensorplay {
namespace cpu {

using namespace tensorplay::parallel;

namespace {

template <typename T>
struct SparseBlasTypeTag { using type = T; };

template <typename F>
void dispatch_sampled_dtype(DType dtype, F&& f) {
    // The op contract admits float/double/complex only; keeping the switch
    // narrow also avoids instantiating the vector dot loop for integer or
    // boolean lanes.
    switch (dtype) {
        case DType::Float32: f(SparseBlasTypeTag<float>{}); return;
        case DType::Float64: f(SparseBlasTypeTag<double>{}); return;
        case DType::ComplexFloat:
            f(SparseBlasTypeTag<complex<float>>{});
            return;
        case DType::ComplexDouble:
            f(SparseBlasTypeTag<complex<double>>{});
            return;
        default:
            TP_THROW(NotImplementedError,
                     "unsupported dtype in sparse_sampled_addmm");
    }
}

template <typename F>
void dispatch_sampled_index(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Int32: f(SparseBlasTypeTag<int32_t>{}); return;
        case DType::Int64: f(SparseBlasTypeTag<int64_t>{}); return;
        default:
            TP_THROW(TypeError,
                     "sparse_sampled_addmm: CSR indices must be Int32 or Int64");
    }
}

// Dot product of two length-K contiguous vectors.  Floating types use the
// vector register; complex and other unsupported lanes fall back to scalar.
template <typename scalar_t>
inline scalar_t vector_dot(const scalar_t* a, const scalar_t* b, int64_t k) {
    if constexpr (is_complex_type_v<scalar_t>) {
        scalar_t acc = scalar_t(0);
        for (int64_t d = 0; d < k; ++d) acc += a[d] * b[d];
        return acc;
    } else {
        using Vec = tensorplay::vec::Vectorized<scalar_t>;
        const int64_t vec_size = Vec::size();
        Vec acc(static_cast<scalar_t>(0));
        int64_t d = 0;
        for (; d + vec_size <= k; d += vec_size) {
            acc = acc + Vec::loadu(a + d) * Vec::loadu(b + d);
        }
        scalar_t result = acc.reduce_add();
        for (; d < k; ++d) result += a[d] * b[d];
        return result;
    }
}

template <typename scalar_t, typename index_t>
void sampled_addmm_kernel_impl(const Tensor& mat1, const Tensor& mat2_t,
                               const Scalar& beta, const Scalar& alpha,
                               const Tensor& result) {
    // Stored elements per batch slice (values are (*batch, nnz, ...)).
    const int64_t nnz = result._values().size(-1);
    if (nnz == 0) {
        return;
    }
    const int64_t batch = mat1.dim() == 2 ? 1 : mat1.size(0);
    const int64_t M = mat1.size(-2);
    const int64_t K = mat1.size(-1);
    const int64_t N = mat2_t.size(-2);

    const scalar_t beta_v = beta.to<scalar_t>();
    const scalar_t alpha_v = alpha.to<scalar_t>();
    const scalar_t* mat1_data = mat1.data_ptr<scalar_t>();
    const scalar_t* mat2_data = mat2_t.data_ptr<scalar_t>();

    Tensor crow = result._crow_indices().reshape({-1, M + 1}).contiguous();
    Tensor col = result._col_indices().reshape({-1, nnz}).contiguous();
    Tensor values = result._values().reshape({-1, nnz}).contiguous();

    const index_t* crow_ptr = crow.data_ptr<index_t>();
    const index_t* col_ptr = col.data_ptr<index_t>();
    scalar_t* values_ptr = values.data_ptr<scalar_t>();

    // Rows are the parallel unit.  A stored element (m, n) reads the
    // contiguous mat1 row m and mat2_t row n, so per-row partitioning keeps
    // the shared mat1 row in cache across the row's stored elements.
    const int64_t row_blocks = batch * M;
    parallel_for(0, row_blocks, 1, [&](int64_t begin, int64_t end) {
        for (int64_t rb = begin; rb < end; ++rb) {
            const int64_t b = rb / M;
            const int64_t m = rb % M;
            const scalar_t* mat1_row = mat1_data + b * M * K + m * K;
            const int64_t row_start = crow_ptr[b * (M + 1) + m];
            const int64_t row_end = crow_ptr[b * (M + 1) + m + 1];
            for (int64_t e = row_start; e < row_end; ++e) {
                const int64_t n = col_ptr[b * nnz + e];
                const scalar_t dot = vector_dot<scalar_t>(
                    mat1_row, mat2_data + b * N * K + n * K, K);
                values_ptr[b * nnz + e] = alpha_v * dot + beta_v * values_ptr[b * nnz + e];
            }
        }
    });
}

} // namespace

void sampled_addmm_sparse_csr_kernel(const Tensor& mat1, const Tensor& mat2_t,
                                     const Scalar& beta, const Scalar& alpha,
                                     const Tensor& result) {
    dispatch_sampled_dtype(result._values().dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        dispatch_sampled_index(result._crow_indices().dtype(), [&](auto itag) {
            using index_t = typename decltype(itag)::type;
            sampled_addmm_kernel_impl<scalar_t, index_t>(
                mat1, mat2_t, beta, alpha, result);
        });
    });
}

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(sampled_addmm_stub, &sampled_addmm_sparse_csr_kernel)
#endif

} // namespace cpu
} // namespace tensorplay
