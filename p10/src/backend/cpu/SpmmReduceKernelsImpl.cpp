// ISA-tier SpMM-reduction kernels (sum/mean/amax/amin) and their backward
// loops.  Each capability copy registers its slot on the stubs declared in
// cpu/SpmmReduceKernels.h; the base entry points in SpmmReduceKernels.cpp
// own the CSR/CSC conversions, input checks and op registration.

#include "cpu/SpmmReduceKernels.h"
#include "DType.h"
#include "Exception.h"
#include "Parallel.h"

#include "cpu/vec/vec.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace tensorplay {
namespace cpu {
inline namespace CPU_CAPABILITY {

using namespace tensorplay::parallel;

namespace {

template <typename T>
struct SpmmTypeTag { using type = T; };

template <typename F>
void dispatch_spmm_dtype(DType dtype, F&& f) {
    switch (dtype) {
        // Ordered reductions accept real floating storage.
        case DType::Float32: f(SpmmTypeTag<float>{}); return;
        case DType::Float64: f(SpmmTypeTag<double>{}); return;
        case DType::Float16: f(SpmmTypeTag<tensorplay::Half>{}); return;
        case DType::BFloat16: f(SpmmTypeTag<tensorplay::BFloat16>{}); return;
        default:
            TP_THROW(NotImplementedError,
                     "unsupported dtype in _sparse_mm_reduce_impl");
    }
}

template <typename F>
void dispatch_spmm_index(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Int32: f(SpmmTypeTag<int32_t>{}); return;
        case DType::Int64: f(SpmmTypeTag<int64_t>{}); return;
        default:
            TP_THROW(TypeError,
                     "_sparse_mm_reduce_impl: CSR indices must be Int32 or "
                     "Int64");
    }
}

template <typename scalar_t>
inline scalar_t neg_inf() {
    return -std::numeric_limits<scalar_t>::infinity();
}

template <typename scalar_t>
inline scalar_t pos_inf() {
    return std::numeric_limits<scalar_t>::infinity();
}

// Horizontal sum of a vector accumulator.  The 16-bit vector layers only
// expose store(), so reduced-precision accumulators are drained through a
// stack buffer instead of an intrinsic reduction.
template <typename scalar_t>
inline scalar_t vectorized_reduce_add(const tensorplay::vec::Vectorized<scalar_t>& acc) {
    if constexpr (tensorplay::vec::is_reduced_floating_point_v<scalar_t>) {
        __at_align__ scalar_t buf[tensorplay::vec::Vectorized<scalar_t>::size()];
        acc.store(buf);
        scalar_t s = scalar_t(0);
        for (int64_t i = 0; i < tensorplay::vec::Vectorized<scalar_t>::size(); ++i) {
            s += buf[i];
        }
        return s;
    } else {
        return acc.reduce_add();
    }
}

// Forward reduction for one row into out_ptr (and optionally arg_ptr).
// `reduce`: 0 sum, 1 mean, 2 amax, 3 amin.
template <typename scalar_t, typename index_t, typename Vec>
inline void reduce_row(const scalar_t* other_data, const index_t* col,
                       const scalar_t* values, int64_t row_start,
                       int64_t row_end, int64_t K, int reduce, bool need_arg,
                       int64_t nnz, scalar_t* out_ptr, index_t* arg_ptr) {
    const int64_t count = row_end - row_start;
    if (reduce == 2) {
        for (int64_t k = 0; k < K; ++k) out_ptr[k] = neg_inf<scalar_t>();
    } else if (reduce == 3) {
        for (int64_t k = 0; k < K; ++k) out_ptr[k] = pos_inf<scalar_t>();
    } else {
        for (int64_t k = 0; k < K; ++k) out_ptr[k] = scalar_t(0);
    }
    if (need_arg) {
        for (int64_t k = 0; k < K; ++k) arg_ptr[k] = static_cast<index_t>(nnz);
    }
    if (count == 0) {
        return;
    }

    if (reduce == 0 || reduce == 1) {
        const int64_t vec_size = Vec::size();
        for (int64_t e = row_start; e < row_end; ++e) {
            const int64_t c = col[e];
            const scalar_t v = values[e];
            const scalar_t* other_row = other_data + c * K;
            int64_t k = 0;
            for (; k + vec_size <= K; k += vec_size) {
                const Vec prod = Vec(v) * Vec::loadu(other_row + k);
                (Vec::loadu(out_ptr + k) + prod).store(out_ptr + k);
            }
            for (; k < K; ++k) {
                out_ptr[k] += v * other_row[k];
            }
        }
        if (reduce == 1) {
            const scalar_t inv = scalar_t(1) / static_cast<scalar_t>(count);
            for (int64_t k = 0; k < K; ++k) out_ptr[k] *= inv;
        }
        return;
    }

    // amax/amin: vectorized compare when no arg is needed; scalar with arg
    // tracking when autograd asks for the winning index.
    if (!need_arg) {
        const int64_t vec_size = Vec::size();
        for (int64_t e = row_start; e < row_end; ++e) {
            const int64_t c = col[e];
            const scalar_t v = values[e];
            const scalar_t* other_row = other_data + c * K;
            int64_t k = 0;
            for (; k + vec_size <= K; k += vec_size) {
                const Vec prod = Vec(v) * Vec::loadu(other_row + k);
                const Vec cur = Vec::loadu(out_ptr + k);
                (reduce == 2 ? tensorplay::vec::maximum(cur, prod)
                             : tensorplay::vec::minimum(cur, prod))
                    .store(out_ptr + k);
            }
            for (; k < K; ++k) {
                const scalar_t prod = v * other_row[k];
                if (reduce == 2) {
                    if (prod > out_ptr[k] || !(prod == prod)) out_ptr[k] = prod;
                } else {
                    if (prod < out_ptr[k] || !(prod == prod)) out_ptr[k] = prod;
                }
            }
        }
        return;
    }

    for (int64_t e = row_start; e < row_end; ++e) {
        const int64_t c = col[e];
        const scalar_t v = values[e];
        const scalar_t* other_row = other_data + c * K;
        for (int64_t k = 0; k < K; ++k) {
            const scalar_t prod = v * other_row[k];
            if (reduce == 2) {
                if (prod > out_ptr[k] || !(prod == prod)) {
                    out_ptr[k] = prod;
                    arg_ptr[k] = static_cast<index_t>(e);
                }
            } else {
                if (prod < out_ptr[k] || !(prod == prod)) {
                    out_ptr[k] = prod;
                    arg_ptr[k] = static_cast<index_t>(e);
                }
            }
        }
    }
}

template <typename scalar_t, typename index_t>
void spmm_reduce_kernel_impl(const Tensor& out, const Tensor& arg_out,
                             const Tensor& crow, const Tensor& col,
                             const Tensor& values, const Tensor& other,
                             int reduce, bool need_arg) {
    const int64_t nnz = values.numel();
    const int64_t M = crow.numel() - 1;
    const int64_t K = other.size(1);
    if (M == 0 || K == 0) {
        return;
    }
    scalar_t* out_data = out.data_ptr<scalar_t>();
    index_t* arg_data = need_arg ? arg_out.data_ptr<index_t>() : nullptr;
    const index_t* crow_data = crow.data_ptr<index_t>();
    const index_t* col_data = col.data_ptr<index_t>();
    const scalar_t* values_data = values.data_ptr<scalar_t>();
    const scalar_t* other_data = other.const_data_ptr<scalar_t>();

    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            scalar_t* out_ptr = out_data + m * K;
            index_t* arg_ptr = need_arg ? arg_data + m * K : nullptr;
            reduce_row<scalar_t, index_t, Vec>(
                other_data, col_data, values_data, crow_data[m],
                crow_data[m + 1], K, reduce, need_arg, nnz, out_ptr, arg_ptr);
        }
    });
}

// grad_self for sum/mean: values[e] = dot(other[col[e]], grad_out[row[e]]),
// scaled by 1/row_nnz for mean.
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_input_kernel_impl(
    const Tensor& grad_self, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& other, const Tensor& row,
    int reduce) {
    const int64_t nnz = grad_self._nnz();
    if (nnz == 0) {
        return;
    }
    const int64_t K = grad_out.size(1);
    scalar_t* grad_values_data = grad_self._values().data_ptr<scalar_t>();
    const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
    const index_t* crow_data = crow.data_ptr<index_t>();
    const index_t* col_data = col.data_ptr<index_t>();
    const scalar_t* other_data = other.const_data_ptr<scalar_t>();
    const index_t* row_data = row.data_ptr<index_t>();

    using Vec = tensorplay::vec::Vectorized<scalar_t>;
    parallel_for(0, nnz, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const int64_t r = row_data[i];
            const int64_t c = col_data[i];
            const scalar_t* other_row = other_data + c * K;
            const scalar_t* grad_row = grad_out_data + r * K;
            Vec acc(static_cast<scalar_t>(0));
            int64_t k = 0;
            const int64_t vec_size = Vec::size();
            for (; k + vec_size <= K; k += vec_size) {
                acc = acc + Vec::loadu(other_row + k) * Vec::loadu(grad_row + k);
            }
            scalar_t val = vectorized_reduce_add(acc);
            for (; k < K; ++k) {
                val += other_row[k] * grad_row[k];
            }
            if (reduce == 1) {
                const int64_t count = crow_data[r + 1] - crow_data[r];
                val /= static_cast<scalar_t>(count);
            }
            grad_values_data[i] = val;
        }
    });
}

// grad_self for amax/amin: scatter grad_out-weighted other rows into the
// winning stored elements.
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_input_arg_kernel_impl(
    const Tensor& grad_self, const Tensor& grad_out, const Tensor& col,
    const Tensor& other, const Tensor& arg_out, int64_t nnz) {
    const int64_t M = grad_out.size(0);
    const int64_t K = grad_out.size(1);
    if (M == 0 || K == 0) {
        return;
    }
    scalar_t* grad_values_data = grad_self._values().data_ptr<scalar_t>();
    const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
    const index_t* col_data = col.data_ptr<index_t>();
    const scalar_t* other_data = other.const_data_ptr<scalar_t>();
    const index_t* arg_data = arg_out.const_data_ptr<index_t>();

    Tensor grad = Tensor::empty({M, K}, grad_out.dtype(), grad_out.device());
    scalar_t* grad_data = grad.data_ptr<scalar_t>();

    parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const scalar_t* grad_out_row = grad_out_data + m * K;
            const index_t* arg_row = arg_data + m * K;
            scalar_t* grad_row = grad_data + m * K;
            for (int64_t k = 0; k < K; ++k) {
                const index_t arg = arg_row[k];
                if (arg == static_cast<index_t>(nnz)) {
                    grad_row[k] = scalar_t(0);
                } else {
                    const int64_t c = col_data[arg];
                    grad_row[k] = other_data[c * K + k] * grad_out_row[k];
                }
            }
        }
    });

    // Serial scatter-add of the dense grad into the sparse values.  The
    // winning index is unique per output cell, so this is a pure gather-add.
    for (int64_t i = 0; i < M * K; ++i) {
        const index_t arg = arg_data[i];
        if (arg != static_cast<index_t>(nnz)) {
            grad_values_data[arg] += grad_data[i];
        }
    }
}

// grad_other for sum/mean: column-major accumulation over the CSC form.
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_other_kernel_impl(
    const Tensor& grad_other, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& values, const Tensor& row,
    const Tensor& ccol, const Tensor& permute, int reduce) {
    (void)crow;
    (void)col;
    (void)reduce;
    const int64_t N = ccol.numel() - 1;
    const int64_t K = grad_out.size(1);
    if (N == 0 || K == 0) {
        return;
    }
    scalar_t* grad_other_data = grad_other.data_ptr<scalar_t>();
    const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
    const index_t* ccol_data = ccol.data_ptr<index_t>();
    const index_t* permute_data = permute.data_ptr<index_t>();
    const index_t* row_data = row.data_ptr<index_t>();
    const scalar_t* values_data = values.data_ptr<scalar_t>();

    parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t n = begin; n < end; ++n) {
            scalar_t* grad_other_row = grad_other_data + n * K;
            for (int64_t p = ccol_data[n]; p < ccol_data[n + 1]; ++p) {
                const int64_t e = permute_data[p];
                const int64_t r = row_data[e];
                const scalar_t v = values_data[e];
                const scalar_t* grad_out_row = grad_out_data + r * K;
                for (int64_t k = 0; k < K; ++k) {
                    grad_other_row[k] += v * grad_out_row[k];
                }
            }
        }
    });
}

// grad_other for amax/amin: scatter values*grad_out at the arg positions.
template <typename scalar_t, typename index_t>
void spmm_reduce_backward_other_arg_kernel_impl(
    const Tensor& grad_other, const Tensor& grad_out, const Tensor& col,
    const Tensor& values, const Tensor& arg_out, int64_t nnz) {
    const int64_t M = grad_out.size(0);
    const int64_t K = grad_out.size(1);
    if (M == 0 || K == 0) {
        return;
    }
    scalar_t* grad_other_data = grad_other.data_ptr<scalar_t>();
    const scalar_t* grad_out_data = grad_out.const_data_ptr<scalar_t>();
    const index_t* col_data = col.data_ptr<index_t>();
    const scalar_t* values_data = values.data_ptr<scalar_t>();
    const index_t* arg_data = arg_out.const_data_ptr<index_t>();

    for (int64_t m = 0; m < M; ++m) {
        const scalar_t* grad_out_row = grad_out_data + m * K;
        const index_t* arg_row = arg_data + m * K;
        for (int64_t k = 0; k < K; ++k) {
            const index_t arg = arg_row[k];
            if (arg != static_cast<index_t>(nnz)) {
                const int64_t c = col_data[arg];
                grad_other_data[c * K + k] += values_data[arg] * grad_out_row[k];
            }
        }
    }
}

} // namespace

void spmm_reduce_kernel(const Tensor& out, const Tensor& arg_out,
                        const Tensor& crow, const Tensor& col,
                        const Tensor& values, const Tensor& other,
                        int reduce, bool need_arg) {
    dispatch_spmm_dtype(values.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        dispatch_spmm_index(crow.dtype(), [&](auto itag) {
            using index_t = typename decltype(itag)::type;
            spmm_reduce_kernel_impl<scalar_t, index_t>(
                out, arg_out, crow, col, values, other, reduce, need_arg);
        });
    });
}

void spmm_reduce_backward_input_kernel(
    const Tensor& grad_self, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& other, const Tensor& arg_out,
    const Tensor& row, int reduce) {
    if (reduce == 2 || reduce == 3) {
        dispatch_spmm_dtype(grad_self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            dispatch_spmm_index(col.dtype(), [&](auto itag) {
                using index_t = typename decltype(itag)::type;
                spmm_reduce_backward_input_arg_kernel_impl<scalar_t, index_t>(
                    grad_self, grad_out, col, other, arg_out, grad_self._nnz());
            });
        });
    } else {
        dispatch_spmm_dtype(grad_self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            dispatch_spmm_index(col.dtype(), [&](auto itag) {
                using index_t = typename decltype(itag)::type;
                spmm_reduce_backward_input_kernel_impl<scalar_t, index_t>(
                    grad_self, grad_out, crow, col, other, row, reduce);
            });
        });
    }
}

void spmm_reduce_backward_other_kernel(
    const Tensor& grad_other, const Tensor& grad_out, const Tensor& crow,
    const Tensor& col, const Tensor& values, const Tensor& arg_out,
    const Tensor& row, const Tensor& ccol, const Tensor& permute,
    int reduce) {
    if (reduce == 2 || reduce == 3) {
        dispatch_spmm_dtype(grad_other.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            dispatch_spmm_index(col.dtype(), [&](auto itag) {
                using index_t = typename decltype(itag)::type;
                spmm_reduce_backward_other_arg_kernel_impl<scalar_t, index_t>(
                    grad_other, grad_out, col, values, arg_out,
                    values.numel());
            });
        });
    } else {
        dispatch_spmm_dtype(grad_other.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            dispatch_spmm_index(col.dtype(), [&](auto itag) {
                using index_t = typename decltype(itag)::type;
                spmm_reduce_backward_other_kernel_impl<scalar_t, index_t>(
                    grad_other, grad_out, crow, col, values, row, ccol,
                    permute, reduce);
            });
        });
    }
}

#ifndef CPU_CAPABILITY_AVX512
REGISTER_DISPATCH(spmm_reduce_stub, &spmm_reduce_kernel)
REGISTER_DISPATCH(spmm_reduce_backward_input_stub,
                  &spmm_reduce_backward_input_kernel)
REGISTER_DISPATCH(spmm_reduce_backward_other_stub,
                  &spmm_reduce_backward_other_kernel)
#else
ALSO_REGISTER_AVX512_DISPATCH(spmm_reduce_stub, &spmm_reduce_kernel)
ALSO_REGISTER_AVX512_DISPATCH(spmm_reduce_backward_input_stub,
                             &spmm_reduce_backward_input_kernel)
ALSO_REGISTER_AVX512_DISPATCH(spmm_reduce_backward_other_stub,
                             &spmm_reduce_backward_other_kernel)
#endif

} // namespace CPU_CAPABILITY
} // namespace cpu
} // namespace tensorplay
