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
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

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

// Dense-out SpMM with seed: out = beta*self + alpha*mat1@mat2.  `self` is a
// contiguous (M, N) seed (already broadcast by the caller); `dense` is a
// contiguous (K, N) right operand.  Rows are the parallel unit; the vector
// loop accumulates the per-row nonzero contributions into the output row.
// Complex lanes fall back to a scalar loop (no vector registers exist for
// them in the per-ISA tier).
template <typename scalar_t, typename index_t>
void sparse_addmm_kernel_impl(const Tensor& out, const Tensor& self,
                              const Tensor& crow, const Tensor& col,
                              const Tensor& values, const Tensor& dense,
                              const Scalar& beta, const Scalar& alpha) {
    const int64_t M = out.size(0);
    const int64_t N = out.size(1);
    if (M == 0 || N == 0) {
        return;
    }
    const scalar_t beta_v = beta.to<scalar_t>();
    const scalar_t alpha_v = alpha.to<scalar_t>();
    const index_t* crow_data = crow.data_ptr<index_t>();
    const index_t* col_data = col.data_ptr<index_t>();
    const scalar_t* value_data = values.data_ptr<scalar_t>();
    const scalar_t* dense_data = dense.data_ptr<scalar_t>();
    const scalar_t* self_data = self.data_ptr<scalar_t>();
    scalar_t* out_data = out.data_ptr<scalar_t>();

    if constexpr (is_complex_type_v<scalar_t>) {
        parallel_for(0, M, std::max<int64_t>(1, 32768 / N),
                 [&](int64_t begin, int64_t end) {
            for (int64_t m = begin; m < end; ++m) {
                scalar_t* out_row = out_data + m * N;
                const scalar_t* self_row = self_data + m * N;
                for (int64_t j = 0; j < N; ++j) {
                    out_row[j] = beta_v * self_row[j];
                }
                for (int64_t e = crow_data[m]; e < crow_data[m + 1]; ++e) {
                    const scalar_t scale = alpha_v * value_data[e];
                    const scalar_t* dense_row = dense_data + col_data[e] * N;
                    for (int64_t j = 0; j < N; ++j) {
                        out_row[j] += scale * dense_row[j];
                    }
                }
            }
        });
        return;
    }

    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    const int64_t vec_size = Vec::size();

    parallel_for(0, M, std::max<int64_t>(1, 32768 / N),
                 [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            scalar_t* out_row = out_data + m * N;
            const scalar_t* self_row = self_data + m * N;
            const int64_t row_start = crow_data[m];
            const int64_t row_end = crow_data[m + 1];
            if (row_start == row_end) {
                int64_t j = 0;
                for (; j + vec_size <= N; j += vec_size) {
                    (Vec::loadu(self_row + j) * Vec(beta_v)).store(out_row + j);
                }
                for (; j < N; ++j) {
                    out_row[j] = beta_v * self_row[j];
                }
                continue;
            }
            const scalar_t scale0 = alpha_v * value_data[row_start];
            const scalar_t* dense0 = dense_data + col_data[row_start] * N;
            int64_t j = 0;
            for (; j + vec_size <= N; j += vec_size) {
                (Vec::loadu(self_row + j) * Vec(beta_v) +
                 Vec(scale0) * Vec::loadu(dense0 + j))
                    .store(out_row + j);
            }
            for (; j < N; ++j) {
                out_row[j] = beta_v * self_row[j] + scale0 * dense0[j];
            }

            // Process the remaining nonzeros in groups of four.  Fusing four
            // sparse contributions per output pass quarters the L1 traffic on
            // the output row and keeps four dense loads in flight.  The
            // next group's leading dense row is prefetched while the current
            // group is consumed, hiding the L3 latency of row switches.
            int64_t e = row_start + 1;
            for (; e + 3 < row_end; e += 4) {
                const scalar_t s0 = alpha_v * value_data[e];
                const scalar_t s1 = alpha_v * value_data[e + 1];
                const scalar_t s2 = alpha_v * value_data[e + 2];
                const scalar_t s3 = alpha_v * value_data[e + 3];
                const scalar_t* d0 = dense_data + col_data[e] * N;
                const scalar_t* d1 = dense_data + col_data[e + 1] * N;
                const scalar_t* d2 = dense_data + col_data[e + 2] * N;
                const scalar_t* d3 = dense_data + col_data[e + 3] * N;
#if defined(__x86_64__) || defined(__i386__)
                if (e + 4 < row_end) {
                    _mm_prefetch(
                        reinterpret_cast<const char*>(
                            dense_data + col_data[e + 4] * N),
                        _MM_HINT_T0);
                }
#endif
                int64_t k = 0;
                for (; k + vec_size <= N; k += vec_size) {
                    Vec acc = Vec::loadu(out_row + k);
                    acc = acc + Vec(s0) * Vec::loadu(d0 + k);
                    acc = acc + Vec(s1) * Vec::loadu(d1 + k);
                    acc = acc + Vec(s2) * Vec::loadu(d2 + k);
                    acc = acc + Vec(s3) * Vec::loadu(d3 + k);
                    acc.store(out_row + k);
                }
                for (; k < N; ++k) {
                    out_row[k] += s0 * d0[k] + s1 * d1[k] +
                                  s2 * d2[k] + s3 * d3[k];
                }
            }
            for (; e + 1 < row_end; e += 2) {
                const scalar_t s0 = alpha_v * value_data[e];
                const scalar_t s1 = alpha_v * value_data[e + 1];
                const scalar_t* d0 = dense_data + col_data[e] * N;
                const scalar_t* d1 = dense_data + col_data[e + 1] * N;
                int64_t k = 0;
                for (; k + vec_size <= N; k += vec_size) {
                    Vec acc = Vec::loadu(out_row + k);
                    acc = acc + Vec(s0) * Vec::loadu(d0 + k);
                    acc = acc + Vec(s1) * Vec::loadu(d1 + k);
                    acc.store(out_row + k);
                }
                for (; k < N; ++k) {
                    out_row[k] += s0 * d0[k] + s1 * d1[k];
                }
            }
            if (e < row_end) {
                const scalar_t s = alpha_v * value_data[e];
                const scalar_t* d = dense_data + col_data[e] * N;
                int64_t k = 0;
                for (; k + vec_size <= N; k += vec_size) {
                    (Vec::loadu(out_row + k) + Vec(s) * Vec::loadu(d + k))
                        .store(out_row + k);
                }
                for (; k < N; ++k) {
                    out_row[k] += s * d[k];
                }
            }
        }
    });
}

} // namespace

void sparse_addmm_sparse_csr_kernel(const Tensor& out, const Tensor& self,
                                    const Tensor& crow, const Tensor& col,
                                    const Tensor& values, const Tensor& dense,
                                    const Scalar& beta, const Scalar& alpha) {
    dispatch_sampled_dtype(values.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        dispatch_sampled_index(crow.dtype(), [&](auto itag) {
            using index_t = typename decltype(itag)::type;
            sparse_addmm_kernel_impl<scalar_t, index_t>(
                out, self, crow, col, values, dense, beta, alpha);
        });
    });
}

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
REGISTER_DISPATCH(sparse_addmm_stub, &sparse_addmm_sparse_csr_kernel)
#else
ALSO_REGISTER_AVX512_DISPATCH(sampled_addmm_stub, &sampled_addmm_sparse_csr_kernel)
ALSO_REGISTER_AVX512_DISPATCH(sparse_addmm_stub, &sparse_addmm_sparse_csr_kernel)
#endif

} // namespace CPU_CAPABILITY
} // namespace cpu
} // namespace tensorplay
