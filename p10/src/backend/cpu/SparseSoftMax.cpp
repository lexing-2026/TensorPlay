// Sparse COO softmax / log_softmax and their backward data kernels.
//
// The reduction runs over independent "pools" of stored entries: entries
// whose coordinates agree outside the softmax dim share one softmax
// computation.  Unspecified entries contribute nothing to the exponent sums
// (they behave as negative infinities), so the output keeps the input's
// coordinates and values count.  When the softmax dim lies in the dense part
// the values payload reduces with the dense kernels directly.
//
// The backward routes the output gradient back per pool:
//   softmax:     gI_i = out_i * (g_i - sum_j out_j * g_j)
//   log_softmax: gI_i = g_i - exp(out_i) * sum_j g_j
// with the pool sums accumulated over matching grad coordinates (matched by
// flattened-coordinate binary search).

#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Parallel.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace tensorplay {
namespace cpu {

namespace {

// Flattened dense offsets of the stored entries; `dim >= 0` collapses that
// dimension's coordinate to zero first.
std::vector<int64_t> sparse_offsets(const Tensor& indices,
                                    const std::vector<int64_t>& sizes,
                                    int64_t dim) {
    const int64_t ndim = indices.size(0);
    const int64_t nnz = indices.size(1);
    std::vector<int64_t> strides(static_cast<size_t>(ndim), 1);
    const int64_t* index_data = indices.data_ptr<int64_t>();
    if (ndim > 1) {
        for (int64_t i = ndim - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] =
                strides[static_cast<size_t>(i + 1)] *
                (i + 1 == dim ? 1 : sizes[static_cast<size_t>(i + 1)]);
        }
    }
    std::vector<int64_t> offsets(static_cast<size_t>(nnz), 0);
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t acc = 0;
        for (int64_t j = 0; j < ndim; ++j) {
            if (j == dim) continue;
            acc += strides[static_cast<size_t>(j)] *
                   index_data[j * nnz + i];
        }
        offsets[static_cast<size_t>(i)] = acc;
    }
    return offsets;
}

// Pool index per stored entry: entries sharing every coordinate outside the
// softmax dim land in the same pool.
std::vector<std::vector<int64_t>> sparse_pools(
    const Tensor& indices, const std::vector<int64_t>& sizes, int64_t dim) {
    const int64_t ndim = indices.size(0);
    const int64_t nnz = indices.size(1);
    std::vector<int64_t> strides(static_cast<size_t>(ndim), 1);
    const int64_t* index_data = indices.data_ptr<int64_t>();
    if (ndim > 1) {
        for (int64_t i = ndim - 2; i >= 0; --i) {
            strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] *
                (i + 1 == dim ? 1 : sizes[static_cast<size_t>(i + 1)]);
        }
    }
    std::vector<std::vector<int64_t>> pools;
    for (int64_t i = 0; i < nnz; ++i) {
        int64_t pool_index = 0;
        for (int64_t j = 0; j < ndim; ++j) {
            if (j == dim) continue;
            pool_index += strides[static_cast<size_t>(j)] *
                          index_data[j * nnz + i];
        }
        if (static_cast<int64_t>(pools.size()) <= pool_index) {
            pools.resize(static_cast<size_t>(pool_index) + 1);
        }
        pools[static_cast<size_t>(pool_index)].push_back(i);
    }
    return pools;
}

template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_float_dtype(DType dtype, F&& f) {
    switch (dtype) {
        case DType::Float32: f(TypeTag<float>{}); return;
        case DType::Float64: f(TypeTag<double>{}); return;
        default:
            TP_THROW(NotImplementedError,
                     "sparse softmax: unsupported dtype");
    }
}

// Shared preprocessing: coalesce, build the output shell with the same
// coordinates, wrap the dim.
std::tuple<Tensor, Tensor, int64_t> softmax_sparse_preprocessing(
    const Tensor& input_, int64_t dim_, const char* fn_name) {
    TP_CHECK(input_.is_sparse() && !input_.is_sparse_compressed(),
             fn_name, ": expected a sparse COO tensor");
    Tensor input = input_.is_coalesced() ? input_ : input_.coalesce();
    Tensor indices = input._indices().contiguous();
    Tensor output = Tensor::make_sparse_coo_tensor(
        indices,
        Tensor::empty(static_cast<std::vector<int64_t>>(input._values().shape()),
                      input.dtype(), input.device()),
        static_cast<std::vector<int64_t>>(input.shape()), true);
    const int64_t ndim = input.dim();
    TP_CHECK(dim_ >= -ndim && dim_ < ndim,
             "Dimension out of range (expected to be in range of [", -ndim,
             ", ", ndim - 1, "], but got ", dim_, ")");
    const int64_t dim = dim_ < 0 ? dim_ + ndim : dim_;
    return {input, output, dim};
}

Tensor ops_softmax(const Tensor& values, int64_t dim) {
    return tensorplay::tpx::ops::_softmax(values, dim, false);
}
Tensor ops_log_softmax(const Tensor& values, int64_t dim) {
    return tensorplay::tpx::ops::_log_softmax(values, dim, false);
}
Tensor ops_softmax_backward(const Tensor& grad, const Tensor& output,
                            int64_t dim) {
    return tensorplay::tpx::ops::_softmax_backward_data(
        grad, output, dim, output.dtype());
}
Tensor ops_log_softmax_backward(const Tensor& grad, const Tensor& output,
                                int64_t dim) {
    return tensorplay::tpx::ops::_log_softmax_backward_data(
        grad, output, dim, output.dtype());
}

template <typename scalar_t, bool LogSoftMax>
void sparse_coo_softmax(Tensor output, const Tensor& input, int64_t dim) {
    const int64_t sparse_dim = input.sparse_dim();
    Tensor indices = input._indices().contiguous();
    Tensor values = input._values().contiguous();
    Tensor out_indices = output._indices();
    Tensor out_values = output._values();
    out_indices.copy_(indices);

    auto sizes = static_cast<std::vector<int64_t>>(input.shape());
    const int64_t nnz = values.size(0);

    if (dim >= sparse_dim) {
        // The softmax dim is inside the dense payload: reduce the values
        // with the dense kernels along the payload-relative dim.
        const int64_t values_dim = dim - sparse_dim + 1;
        Tensor new_values = LogSoftMax
            ? ops_log_softmax(values, values_dim)
            : ops_softmax(values, values_dim);
        out_values.copy_(new_values);
        return;
    }

    const int64_t nvalues = [&] {
        int64_t acc = 1;
        for (int64_t d = sparse_dim; d < static_cast<int64_t>(sizes.size()); ++d) {
            acc *= sizes[static_cast<size_t>(d)];
        }
        return acc;
    }();

    scalar_t* out_ptr = out_values.data_ptr<scalar_t>();
    const scalar_t* values_ptr = values.data_ptr<scalar_t>();
    auto pools = sparse_pools(indices, sizes, dim);

    tensorplay::parallel::parallel_for(
        0, static_cast<int64_t>(pools.size()), 1,
        [&](int64_t begin, int64_t end) {
            for (int64_t p = begin; p < end; ++p) {
                const auto& pool_indices = pools[static_cast<size_t>(p)];
                if (pool_indices.empty()) continue;
                std::vector<scalar_t> mx_row(static_cast<size_t>(nvalues),
                                             -std::numeric_limits<scalar_t>::infinity());
                std::vector<scalar_t> exp_sums_row(static_cast<size_t>(nvalues), 0);

                // Pool maximum.
                for (int64_t i : pool_indices) {
                    const scalar_t* row = values_ptr + i * nvalues;
                    for (int64_t j = 0; j < nvalues; ++j) {
                        mx_row[static_cast<size_t>(j)] =
                            std::max(mx_row[static_cast<size_t>(j)], row[j]);
                    }
                }
                // exp(v - mx) and its sums.
                for (int64_t i : pool_indices) {
                    const scalar_t* row = values_ptr + i * nvalues;
                    scalar_t* out_row = out_ptr + i * nvalues;
                    for (int64_t j = 0; j < nvalues; ++j) {
                        const scalar_t v = std::exp(row[j] - mx_row[static_cast<size_t>(j)]);
                        if (!LogSoftMax) out_row[j] = v;
                        exp_sums_row[static_cast<size_t>(j)] += v;
                    }
                }
                for (int64_t j = 0; j < nvalues; ++j) {
                    if (LogSoftMax) {
                        mx_row[static_cast<size_t>(j)] +=
                            std::log(exp_sums_row[static_cast<size_t>(j)]);
                    } else {
                        exp_sums_row[static_cast<size_t>(j)] =
                            scalar_t(1) / exp_sums_row[static_cast<size_t>(j)];
                    }
                }
                // Normalize (log_softmax: subtract the log-sum-exp).
                for (int64_t i : pool_indices) {
                    const scalar_t* row = values_ptr + i * nvalues;
                    scalar_t* out_row = out_ptr + i * nvalues;
                    for (int64_t j = 0; j < nvalues; ++j) {
                        if (LogSoftMax) {
                            out_row[j] = row[j] - mx_row[static_cast<size_t>(j)];
                        } else {
                            out_row[j] *= exp_sums_row[static_cast<size_t>(j)];
                        }
                    }
                }
            }
        });
}

template <typename scalar_t, bool LogSoftMax>
void sparse_coo_softmax_backward(Tensor grad_input, const Tensor& grad,
                                 const Tensor& output, int64_t dim) {
    const int64_t sparse_dim = output.sparse_dim();
    auto sizes = static_cast<std::vector<int64_t>>(output.shape());
    Tensor grad_indices = grad._indices().contiguous();
    Tensor grad_values = grad._values().contiguous();
    Tensor out_indices = output._indices().contiguous();
    Tensor out_values = output._values().contiguous();
    Tensor values = grad_input._values();
    Tensor indices = grad_input._indices();
    const int64_t out_nnz = out_values.size(0);
    const int64_t grad_nnz = grad_values.size(0);

    values.resize_as_(out_values).zero_();
    indices.copy_(out_indices);

    auto out_offsets = sparse_offsets(out_indices, sizes, -1);
    auto grad_offsets = sparse_offsets(grad_indices, sizes, -1);

    if (dim >= sparse_dim) {
        // Dense payload case: grad and output must share coordinates.
        bool offsets_match = out_offsets == grad_offsets;
        if (offsets_match) {
            Tensor r = LogSoftMax
                ? ops_log_softmax_backward(grad_values, out_values,
                                           dim - sparse_dim + 1)
                : ops_softmax_backward(grad_values, out_values,
                                       dim - sparse_dim + 1);
            values.copy_(r);
        } else {
            for (int64_t i = 0; i < out_nnz; ++i) {
                auto low = std::lower_bound(grad_offsets.begin(),
                                            grad_offsets.end(),
                                            out_offsets[static_cast<size_t>(i)]);
                const int64_t j = low - grad_offsets.begin();
                if (j < grad_nnz &&
                    out_offsets[static_cast<size_t>(i)] ==
                        grad_offsets[static_cast<size_t>(j)]) {
                    Tensor r = LogSoftMax
                        ? ops_log_softmax_backward(
                            grad_values.select(0, j),
                            out_values.select(0, i), dim - sparse_dim)
                        : ops_softmax_backward(
                            grad_values.select(0, j),
                            out_values.select(0, i), dim - sparse_dim);
                    values.select(0, i).copy_(r);
                }
            }
        }
        return;
    }

    const int64_t nvalues = [&] {
        int64_t acc = 1;
        for (int64_t d = sparse_dim; d < static_cast<int64_t>(sizes.size()); ++d) {
            acc *= sizes[static_cast<size_t>(d)];
        }
        return acc;
    }();
    const int64_t nnz = values.size(0);

    scalar_t* values_ptr = values.data_ptr<scalar_t>();
    const scalar_t* out_ptr = out_values.data_ptr<scalar_t>();
    const scalar_t* grad_ptr = grad_values.data_ptr<scalar_t>();

    auto pools = sparse_pools(out_indices, sizes, dim);
    tensorplay::parallel::parallel_for(
        0, static_cast<int64_t>(pools.size()), 1,
        [&](int64_t begin, int64_t end) {
            for (int64_t p = begin; p < end; ++p) {
                const auto& pool_indices = pools[static_cast<size_t>(p)];
                if (pool_indices.empty()) continue;
                std::vector<scalar_t> tmp_row(static_cast<size_t>(nvalues), 0);
                // tmp = -sum_j out_j * g_j (softmax) or -sum_j g_j (log).
                for (int64_t i : pool_indices) {
                    const scalar_t* out_row = out_ptr + i * nvalues;
                    auto low = std::lower_bound(grad_offsets.begin(),
                                                grad_offsets.end(),
                                                out_offsets[static_cast<size_t>(i)]);
                    const int64_t j = low - grad_offsets.begin();
                    if (j < grad_nnz && out_offsets[static_cast<size_t>(i)] ==
                                            grad_offsets[static_cast<size_t>(j)]) {
                        const scalar_t* grad_row = grad_ptr + j * nvalues;
                        for (int64_t k = 0; k < nvalues; ++k) {
                            if (LogSoftMax) {
                                tmp_row[static_cast<size_t>(k)] -= grad_row[k];
                            } else {
                                tmp_row[static_cast<size_t>(k)] -=
                                    out_row[k] * grad_row[k];
                            }
                        }
                    }
                }
                // gI = out * (g + tmp) (softmax) or g + exp(out)*tmp (log).
                for (int64_t i : pool_indices) {
                    const scalar_t* out_row = out_ptr + i * nvalues;
                    scalar_t* values_row = values_ptr + i * nvalues;
                    auto low = std::lower_bound(grad_offsets.begin(),
                                                grad_offsets.end(),
                                                out_offsets[static_cast<size_t>(i)]);
                    const int64_t j = low - grad_offsets.begin();
                    if (j < grad_nnz && out_offsets[static_cast<size_t>(i)] ==
                                            grad_offsets[static_cast<size_t>(j)]) {
                        const scalar_t* grad_row = grad_ptr + j * nvalues;
                        for (int64_t k = 0; k < nvalues; ++k) {
                            if (LogSoftMax) {
                                values_row[k] = grad_row[k] +
                                    std::exp(out_row[k]) * tmp_row[static_cast<size_t>(k)];
                            } else {
                                values_row[k] = out_row[k] *
                                    (grad_row[k] + tmp_row[static_cast<size_t>(k)]);
                            }
                        }
                    } else {
                        for (int64_t k = 0; k < nvalues; ++k) {
                            if (LogSoftMax) {
                                values_row[k] = std::exp(out_row[k]) *
                                    tmp_row[static_cast<size_t>(k)];
                            } else {
                                values_row[k] = out_row[k] * tmp_row[static_cast<size_t>(k)];
                            }
                        }
                    }
                }
            }
        });
}

template <bool LogSoftMax>
Tensor softmax_sparse_forward(const Tensor& input_, int64_t dim_,
                              bool half_to_float, const char* fn_name) {
    TP_CHECK(!half_to_float, fn_name,
             ": with half to float conversion is not supported on CPU");
    auto [input, output, dim] =
        softmax_sparse_preprocessing(input_, dim_, fn_name);
    if (input.numel() == 0) {
        return output;
    }
    dispatch_float_dtype(input.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        sparse_coo_softmax<scalar_t, LogSoftMax>(output, input, dim);
    });
    return output;
}

template <bool LogSoftMax>
Tensor softmax_backward_sparse(const Tensor& grad_, const Tensor& output_,
                               int64_t dim_, const Tensor& input_) {
    (void)input_;
    TP_CHECK(grad_.is_sparse() && !grad_.is_sparse_compressed(),
             "_sparse_softmax_backward_data: expected a sparse COO grad");
    TP_CHECK(output_.is_sparse() && !output_.is_sparse_compressed(),
             "_sparse_softmax_backward_data: expected a sparse COO output");
    TP_CHECK(output_.shape() == grad_.shape(),
             "_sparse_softmax_backward_data: grad and output must have the "
             "same sizes");
    int64_t dim = dim_;
    const int64_t ndim = grad_.dim();
    TP_CHECK(dim >= -ndim && dim < ndim,
             "Dimension out of range (expected to be in range of [", -ndim,
             ", ", ndim - 1, "], but got ", dim, ")");
    if (dim < 0) dim += ndim;

    Tensor grad = grad_.is_coalesced() ? grad_ : grad_.coalesce();
    Tensor output = output_.is_coalesced() ? output_ : output_.coalesce();
    TP_CHECK(grad.sparse_dim() == output.sparse_dim(),
             "_sparse_softmax_backward_data: grad and output sparse dimensions "
             "must be equal");
    Tensor grad_input = Tensor::make_sparse_coo_tensor(
        output._indices().contiguous(),
        Tensor::empty(static_cast<std::vector<int64_t>>(output._values().shape()),
                      output.dtype(), output.device()),
        static_cast<std::vector<int64_t>>(output.shape()), true);
    if (output.numel() == 0) {
        return grad_input;
    }
    dispatch_float_dtype(grad.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        sparse_coo_softmax_backward<scalar_t, LogSoftMax>(
            grad_input, grad, output, dim);
    });
    return grad_input;
}

}  // namespace

Tensor _sparse_softmax_cpu(const Tensor& input, int64_t dim, bool half_to_float) {
    return softmax_sparse_forward<false>(input, dim, half_to_float,
                                         "softmax");
}

Tensor _sparse_softmax_int_cpu(const Tensor& input, int64_t dim,
                               std::optional<DType> dtype) {
    Tensor converted = dtype.has_value() && *dtype != DType::Undefined &&
                               *dtype != input.dtype()
                           ? input.to(*dtype)
                           : input;
    return _sparse_softmax_cpu(converted, dim, false);
}

Tensor _sparse_log_softmax_cpu(const Tensor& input, int64_t dim, bool half_to_float) {
    return softmax_sparse_forward<true>(input, dim, half_to_float,
                                        "log_softmax");
}

Tensor _sparse_log_softmax_int_cpu(const Tensor& input, int64_t dim,
                                   std::optional<DType> dtype) {
    Tensor converted = dtype.has_value() && *dtype != DType::Undefined &&
                               *dtype != input.dtype()
                           ? input.to(*dtype)
                           : input;
    return _sparse_log_softmax_cpu(converted, dim, false);
}

Tensor _sparse_softmax_backward_data_cpu(const Tensor& grad,
                                         const Tensor& output, int64_t dim,
                                         const Tensor& input) {
    return softmax_backward_sparse<false>(grad, output, dim, input);
}

Tensor _sparse_log_softmax_backward_data_cpu(const Tensor& grad,
                                             const Tensor& output,
                                             int64_t dim,
                                             const Tensor& input) {
    return softmax_backward_sparse<true>(grad, output, dim, input);
}

}  // namespace cpu
}  // namespace tensorplay

TENSORPLAY_LIBRARY_IMPL(CPU, SparseSoftMax) {
    using namespace tensorplay::cpu;
    m.impl("_sparse_softmax", _sparse_softmax_cpu);
    m.impl("_sparse_softmax.int", _sparse_softmax_int_cpu);
    m.impl("_sparse_log_softmax", _sparse_log_softmax_cpu);
    m.impl("_sparse_log_softmax.int", _sparse_log_softmax_int_cpu);
    m.impl("_sparse_softmax_backward_data", _sparse_softmax_backward_data_cpu);
    m.impl("_sparse_log_softmax_backward_data",
           _sparse_log_softmax_backward_data_cpu);
}
