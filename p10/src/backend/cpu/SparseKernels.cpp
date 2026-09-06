#include "SparseKernels.h"
#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>
#include <unordered_set>

namespace tensorplay {
namespace cpu {
namespace {

template <typename T>
struct TypeTag { using type = T; };

template <typename F>
void dispatch_dtype(DType dtype, F&& f) {
#define TP_SPARSE_DISPATCH(ctype, name) \
    case DType::name: f(TypeTag<ctype>{}); return;
    switch (dtype) {
        TENSORPLAY_FORALL_SCALAR_TYPES_WITH_COMPLEX(TP_SPARSE_DISPATCH)
        default:
            TP_THROW(NotImplementedError, "unsupported dtype in sparse COO kernel");
    }
#undef TP_SPARSE_DISPATCH
}

struct Entry {
    std::vector<int64_t> coordinate;
    int64_t source;
};

bool coordinate_less(const Entry& lhs, const Entry& rhs) {
    return std::lexicographical_compare(lhs.coordinate.begin(), lhs.coordinate.end(),
                                        rhs.coordinate.begin(), rhs.coordinate.end());
}

bool coordinate_equal(const Entry& lhs, const Entry& rhs) {
    return lhs.coordinate == rhs.coordinate;
}

int64_t product(const std::vector<int64_t>& dims) {
    int64_t result = 1;
    for (int64_t dim : dims) result *= dim;
    return result;
}

std::vector<int64_t> dense_shape_for(const Tensor& sparse) {
    const int64_t sparse_dim = sparse.sparse_dim();
    std::vector<int64_t> shape = static_cast<std::vector<int64_t>>(sparse.shape());
    return std::vector<int64_t>(shape.begin() + sparse_dim, shape.end());
}

} // namespace

Tensor sparse_coo_tensor_cpu(const Tensor& indices, const Tensor& values,
                             std::optional<std::vector<int64_t>> size,
                             bool is_coalesced) {
    // and the values shape (dense dims).  The reduction needs host data, so
    // CUDA indices are staged through the CPU like coalesce_cuda does.
    if (!size.has_value()) {
        Tensor host_indices = indices.device().is_cpu()
            ? indices.contiguous() : indices.to(Device(DeviceType::CPU));
        Tensor canonical = host_indices.dtype() == DType::Int64
            ? host_indices : host_indices.to(DType::Int64);
        const int64_t sparse_dim = canonical.size(0);
        const int64_t nnz = canonical.size(1);
        std::vector<int64_t> inferred(static_cast<size_t>(sparse_dim), 0);
        const int64_t* index_data = canonical.data_ptr<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            int64_t max_coordinate = -1;
            for (int64_t n = 0; n < nnz; ++n) {
                max_coordinate = std::max(max_coordinate, index_data[d * nnz + n]);
            }
            inferred[static_cast<size_t>(d)] = max_coordinate + 1;
        }
        for (int64_t i = 1; i < values.dim(); ++i) {
            inferred.push_back(values.size(i));
        }
        size = std::move(inferred);
    }
    return Tensor::make_sparse_coo_tensor(indices, values, *size, is_coalesced);
}

Tensor coalesce_sparse_cpu(const Tensor& self) {
    if (!self.is_sparse()) TP_THROW(RuntimeError, "coalesce(): expected a sparse COO tensor");
    if (self.is_coalesced()) return self;

    Tensor indices = self._indices().contiguous();
    Tensor values = self._values().contiguous();
    const int64_t sparse_dim = self.sparse_dim();
    const int64_t nnz = indices.size(1);
    const std::vector<int64_t> dense_shape = dense_shape_for(self);
    const int64_t dense_numel = product(dense_shape);

    std::vector<Entry> entries;
    entries.reserve(static_cast<size_t>(nnz));
    const int64_t* index_data = indices.data_ptr<int64_t>();
    for (int64_t n = 0; n < nnz; ++n) {
        Entry entry;
        entry.source = n;
        entry.coordinate.resize(static_cast<size_t>(sparse_dim));
        for (int64_t d = 0; d < sparse_dim; ++d) {
            const int64_t coordinate = index_data[d * nnz + n];
            if (coordinate < 0 || coordinate >= self.size(d)) {
                TP_THROW(IndexError, "coalesce(): sparse index is out of bounds");
            }
            entry.coordinate[static_cast<size_t>(d)] = coordinate;
        }
        entries.push_back(std::move(entry));
    }
    std::stable_sort(entries.begin(), entries.end(), coordinate_less);

    int64_t output_nnz = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i == 0 || entries[i].coordinate != entries[i - 1].coordinate) ++output_nnz;
    }

    Tensor output_indices = Tensor::empty({sparse_dim, output_nnz}, DType::Int64, self.device());
    std::vector<int64_t> output_values_shape;
    output_values_shape.push_back(output_nnz);
    output_values_shape.insert(output_values_shape.end(), dense_shape.begin(), dense_shape.end());
    Tensor output_values = Tensor::zeros(output_values_shape, self.dtype(), self.device());
    int64_t* output_index_data = output_indices.data_ptr<int64_t>();

    dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = values.data_ptr<scalar_t>();
        scalar_t* destination = output_values.data_ptr<scalar_t>();
        int64_t output_index = -1;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i == 0 || entries[i].coordinate != entries[i - 1].coordinate) {
                ++output_index;
                for (int64_t d = 0; d < sparse_dim; ++d) {
                    output_index_data[d * output_nnz + output_index] =
                        entries[i].coordinate[static_cast<size_t>(d)];
                }
            }
            const scalar_t* source_row = source + entries[i].source * dense_numel;
            scalar_t* destination_row = destination + output_index * dense_numel;
            for (int64_t j = 0; j < dense_numel; ++j) {
                destination_row[j] += source_row[j];
            }
        }
    });

    return Tensor::make_sparse_coo_tensor(
        output_indices, output_values,
        static_cast<std::vector<int64_t>>(self.shape()), true);
}

Tensor sparse_mask_cpu(const Tensor& dense, const Tensor& mask) {
    if (!mask.is_sparse()) TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse");
    if (dense.device() != mask.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mask(): dense and mask must be on the same device");
    }
    if (dense.shape() != mask.shape()) {
        TP_THROW(RuntimeError,
                 "sparse_mask(): operands have incompatible sizes; self and mask must have the same shape");
    }
    Tensor coo_mask = mask.is_sparse_csr() ? to_sparse_coo_cpu(mask) : mask;
    // It does not coalesce an uncoalesced mask or change its duplicate/order
    // semantics.  SparseAdam passes a coalesced gradient when it needs the
    // canonical form explicitly.
    Tensor indices = coo_mask._indices().contiguous();
    const int64_t sparse_dim = coo_mask.sparse_dim();
    const int64_t nnz = indices.size(1);
    const std::vector<int64_t> dense_shape = dense_shape_for(coo_mask);
    const int64_t dense_numel = product(dense_shape);
    Tensor dense_contiguous = dense.is_contiguous() ? dense : dense.contiguous();

    std::vector<int64_t> values_shape;
    values_shape.push_back(nnz);
    values_shape.insert(values_shape.end(), dense_shape.begin(), dense_shape.end());
    Tensor values = Tensor::empty(values_shape, dense.dtype(), dense.device());

    const int64_t* index_data = indices.data_ptr<int64_t>();
    const std::vector<int64_t> dense_strides = dense_contiguous.strides();
    dispatch_dtype(dense.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = dense_contiguous.data_ptr<scalar_t>();
        scalar_t* destination = values.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            int64_t base_offset = 0;
            for (int64_t d = 0; d < sparse_dim; ++d) {
                base_offset += index_data[d * nnz + n] * dense_strides[static_cast<size_t>(d)];
            }
            for (int64_t j = 0; j < dense_numel; ++j) {
                int64_t remainder = j;
                int64_t offset = base_offset;
                for (int64_t d = static_cast<int64_t>(dense_shape.size()) - 1;
                     d >= 0; --d) {
                    const int64_t coordinate = remainder % dense_shape[static_cast<size_t>(d)];
                    remainder /= dense_shape[static_cast<size_t>(d)];
                    const int64_t source_dim = sparse_dim + d;
                    offset += coordinate * dense_strides[static_cast<size_t>(source_dim)];
                }
                destination[n * dense_numel + j] = source[offset];
            }
        }
    });

    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(mask.shape()),
        coo_mask.is_coalesced());
}

Tensor& add_sparse_to_dense_cpu(Tensor& dense, const Tensor& sparse, Scalar alpha) {
    if (dense.is_sparse() || !sparse.is_sparse()) {
        TP_THROW(RuntimeError, "add_: expected a dense self and sparse COO other");
    }
    if (dense.shape() != sparse.shape()) {
        TP_THROW(RuntimeError, "add_: sparse COO operands must have identical sizes");
    }

    Tensor canonical = sparse.is_coalesced() ? sparse : sparse.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values();
    if (values.dtype() != dense.dtype()) {
        values = Tensor::make_sparse_coo_tensor(
            indices, values.to(dense.dtype()),
            static_cast<std::vector<int64_t>>(sparse.shape()), true)._values();
    }
    values = values.contiguous();

    const int64_t sparse_dim = canonical.sparse_dim();
    const int64_t nnz = indices.size(1);
    const std::vector<int64_t> dense_shape = dense_shape_for(canonical);
    const int64_t dense_numel = product(dense_shape);
    const std::vector<int64_t> dense_strides = dense.strides();
    const int64_t* index_data = indices.data_ptr<int64_t>();

    dispatch_dtype(dense.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        scalar_t* destination = dense.data_ptr<scalar_t>();
        const scalar_t* source = values.data_ptr<scalar_t>();
        const scalar_t alpha_value = alpha.to<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            int64_t base_offset = 0;
            for (int64_t d = 0; d < sparse_dim; ++d) {
                const int64_t coordinate = index_data[d * nnz + n];
                if (coordinate < 0 || coordinate >= dense.size(d)) {
                    TP_THROW(IndexError, "add_: sparse index is out of bounds");
                }
                base_offset += coordinate * dense_strides[static_cast<size_t>(d)];
            }
            for (int64_t j = 0; j < dense_numel; ++j) {
                int64_t remainder = j;
                int64_t destination_offset = base_offset;
                for (int64_t d = static_cast<int64_t>(dense_shape.size()) - 1;
                     d >= 0; --d) {
                    const int64_t dim_size = dense_shape[static_cast<size_t>(d)];
                    const int64_t coordinate = dim_size == 0 ? 0 : remainder % dim_size;
                    remainder = dim_size == 0 ? 0 : remainder / dim_size;
                    destination_offset += coordinate *
                        dense_strides[static_cast<size_t>(sparse_dim + d)];
                }
                destination[destination_offset] += alpha_value * source[n * dense_numel + j];
            }
        }
    });
    dense.unsafeGetTensorImpl()->bump_version();
    return dense;
}

Tensor embedding_sparse_backward_cpu(const Tensor& grad,
                                     const Tensor& indices,
                                     int64_t num_weights,
                                     int64_t padding_idx,
                                     bool scale_grad_by_freq) {
    if (scale_grad_by_freq) {
        TP_THROW(RuntimeError,
                 "embedding_backward: scale_grad_by_freq not supported with sparse gradients");
    }
    if (indices.dtype() != DType::Int64 && indices.dtype() != DType::Int32) {
        TP_THROW(TypeError, "embedding_sparse_backward: indices must be Int64 or Int32");
    }
    if (grad.dim() == 0) {
        TP_THROW(RuntimeError, "embedding_sparse_backward: grad must have a feature dimension");
    }
    Tensor index_flat = indices.contiguous().view({indices.numel()});
    Tensor grad_contiguous = grad.contiguous();
    const int64_t num_indices = indices.numel();
    const int64_t row_size = grad.size(grad.dim() - 1);
    if (grad.numel() != num_indices * row_size) {
        TP_THROW(RuntimeError, "embedding_sparse_backward: incompatible grad and indices shapes");
    }

    std::vector<int64_t> selected;
    selected.reserve(static_cast<size_t>(num_indices));
    auto read_index = [&](int64_t i) -> int64_t {
        if (index_flat.dtype() == DType::Int64) return index_flat.data_ptr<int64_t>()[i];
        return static_cast<int64_t>(index_flat.data_ptr<int32_t>()[i]);
    };
    for (int64_t i = 0; i < num_indices; ++i) {
        int64_t index = read_index(i);
        if (index == padding_idx) continue;
        selected.push_back(i);
    }

    Tensor output_indices = Tensor::empty({1, static_cast<int64_t>(selected.size())},
                                          DType::Int64, grad.device());
    std::vector<int64_t> output_values_shape = {
        static_cast<int64_t>(selected.size()), row_size};
    Tensor output_values = Tensor::empty(output_values_shape, grad.dtype(), grad.device());
    int64_t* out_indices = output_indices.data_ptr<int64_t>();
    const int64_t* index64 = index_flat.dtype() == DType::Int64
        ? index_flat.data_ptr<int64_t>() : nullptr;
    const int32_t* index32 = index_flat.dtype() == DType::Int32
        ? index_flat.data_ptr<int32_t>() : nullptr;

    dispatch_dtype(grad.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = grad_contiguous.data_ptr<scalar_t>();
        scalar_t* destination = output_values.data_ptr<scalar_t>();
        for (size_t out = 0; out < selected.size(); ++out) {
            const int64_t source_index = selected[out];
            int64_t index = index64 ? index64[source_index]
                                    : static_cast<int64_t>(index32[source_index]);
            out_indices[out] = index;
            std::copy_n(source + source_index * row_size, row_size,
                        destination + static_cast<int64_t>(out) * row_size);
        }
    });

    const bool coalesced = selected.size() <= 1;
    return Tensor::make_sparse_coo_tensor(output_indices, output_values,
                                          {num_weights, row_size}, coalesced);
}

namespace {

Tensor coo_to_csr_cpu(const Tensor& coalesced, int64_t rows);

Tensor csr_to_coo_cpu(const Tensor& self) {
    if (!self.is_sparse_csr() || self.dim() != 2) {
        TP_THROW(RuntimeError, "CSR to COO conversion requires a 2-D CSR tensor");
    }
    Tensor crow = self._crow_indices().contiguous();
    Tensor col = self._col_indices().contiguous();
    Tensor values = self._values().contiguous();
    if (values.dim() != 1) {
        TP_THROW(RuntimeError,
                 "CSR to COO conversion does not support hybrid values");
    }
    const int64_t rows = self.size(0);
    const int64_t nnz = values.size(0);
    if (crow.size(0) != rows + 1 || col.size(0) != nnz) {
        TP_THROW(RuntimeError, "CSR index buffers do not match the tensor shape");
    }
    Tensor indices = Tensor::empty({2, nnz}, DType::Int64, self.device());
    int64_t* row_data = indices.data_ptr<int64_t>();
    int64_t* col_data = row_data + nnz;
    const int64_t* crow_data = crow.data_ptr<int64_t>();
    const int64_t* source_col_data = col.data_ptr<int64_t>();
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t entry = crow_data[row]; entry < crow_data[row + 1]; ++entry) {
            row_data[entry] = row;
        }
    }
    std::copy_n(source_col_data, nnz, col_data);
    return Tensor::make_sparse_coo_tensor(
        indices, values, static_cast<std::vector<int64_t>>(self.shape()), false);
}

} // namespace

Tensor to_dense_sparse_cpu(const Tensor& self) {
    if (!self.is_sparse()) return self;
    if (self.is_sparse_compressed()) {
        if (self.dim() < 2) {
            TP_THROW(RuntimeError,
                     "to_dense(): compressed tensors need at least 2-D sizes");
        }
        Tensor crow = self._crow_indices().contiguous();
        Tensor col = self._col_indices().contiguous();
        Tensor values = self._values().contiguous();
        const int layout = self.unsafeGetTensorImpl()->sparse_layout();
        const bool row_compressed =
            layout == TensorImpl::kSparseCSRLayout ||
            layout == TensorImpl::kSparseBSRLayout;
        const bool blocked =
            layout == TensorImpl::kSparseBSRLayout ||
            layout == TensorImpl::kSparseBSCLayout;
        const auto bs = self.sparse_blocksize();
        const int64_t b0 = blocked ? bs[0] : 1;
        const int64_t b1 = blocked ? bs[1] : 1;
        const int64_t ndim = self.dim();
        const int64_t n_batch = ndim - 2;
        int64_t batch_count = 1;
        for (int64_t d = 0; d < n_batch; ++d) batch_count *= self.size(d);
        const int64_t rows = self.size(n_batch);
        const int64_t cols = self.size(n_batch + 1);
        const int64_t comp_units = row_compressed ? rows / b0 : cols / b1;
        // Each batch matrix contributes its own compressed/plain components;
        // the flat nnz axis of values/plain splits evenly across batches.
        const int64_t nnz = values.size(0);
        const int64_t nnz_per_batch = batch_count > 0 ? nnz / batch_count : 0;
        const int64_t block_numel = b0 * b1;
        Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());
        dispatch_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            const int64_t* crow_ptr = crow.data_ptr<int64_t>();
            const int64_t* col_ptr = col.data_ptr<int64_t>();
            const scalar_t* value_ptr = values.data_ptr<scalar_t>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            const int64_t matrix_numel = rows * cols;
            for (int64_t k = 0; k < batch_count; ++k) {
                const int64_t* batch_crow = crow_ptr +
                    k * (comp_units + 1);
                const int64_t* batch_col = col_ptr + k * nnz_per_batch;
                const scalar_t* batch_values = value_ptr +
                    k * nnz_per_batch * block_numel;
                scalar_t* batch_out = out_ptr + k * matrix_numel;
                for (int64_t cu = 0; cu < comp_units; ++cu) {
                    for (int64_t t = batch_crow[cu]; t < batch_crow[cu + 1]; ++t) {
                        const int64_t pu = batch_col[t];
                        for (int64_t i = 0; i < b0; ++i) {
                            for (int64_t j = 0; j < b1; ++j) {
                                const int64_t row = row_compressed
                                    ? cu * b0 + i : pu * b0 + i;
                                const int64_t coln = row_compressed
                                    ? pu * b1 + j : cu * b1 + j;
                                batch_out[row * cols + coln] =
                                    batch_values[(t * b0 + i) * b1 + j];
                            }
                        }
                    }
                }
            }
        });
        return out;
    }

    // COO: coalesce first so each coordinate is written exactly once.
    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());
    const int64_t sparse_dim = canonical.sparse_dim();
    const int64_t nnz = indices.size(1);
    std::vector<int64_t> dense_shape = dense_shape_for(canonical);
    const int64_t dense_numel = product(dense_shape);
    const std::vector<int64_t> out_strides = out.strides();

    dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const int64_t* index_data = indices.data_ptr<int64_t>();
        const scalar_t* source = values.data_ptr<scalar_t>();
        scalar_t* destination = out.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            int64_t base_offset = 0;
            for (int64_t d = 0; d < sparse_dim; ++d) {
                base_offset += index_data[d * nnz + n] *
                               out_strides[static_cast<size_t>(d)];
            }
            for (int64_t j = 0; j < dense_numel; ++j) {
                int64_t remainder = j;
                int64_t offset = base_offset;
                for (int64_t d = static_cast<int64_t>(dense_shape.size()) - 1;
                     d >= 0; --d) {
                    const int64_t dim_size = dense_shape[static_cast<size_t>(d)];
                    const int64_t coordinate = remainder % dim_size;
                    remainder /= dim_size;
                    offset += coordinate * out_strides[static_cast<size_t>(sparse_dim + d)];
                }
                destination[offset] += source[n * dense_numel + j];
            }
        }
    });
    return out;
}

int64_t sparse_nnz_cpu(const Tensor& self) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "_nnz(): expected a sparse tensor");
    }
    return self._values().size(0);
}

namespace {

// Positions (row-major flat coordinates) of nonzero elements of a
// contiguous host tensor, used by both dense -> sparse conversions.
std::vector<int64_t> nonzero_positions(const Tensor& contiguous_self) {
    std::vector<int64_t> positions;
    const int64_t numel = contiguous_self.numel();
    dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != scalar_t(0)) positions.push_back(i);
        }
    });
    return positions;
}

Tensor dense_to_sparse_coo_cpu(const Tensor& self, int64_t sparse_dim) {
    const int64_t ndim = self.dim();
    if (sparse_dim < 0 || sparse_dim > ndim) {
        TP_THROW(ValueError,
                 "to_sparse(): sparse_dim must be in [0," +
                     std::to_string(ndim) + "]");
    }
    if (ndim > 0 && sparse_dim == 0) {
        TP_THROW(ValueError,
                 "to_sparse(): sparse_dim must be greater than zero for a non-scalar tensor");
    }

    Tensor contiguous_self = self.contiguous();
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(contiguous_self.shape());
    const int64_t outer_numel =
        product(std::vector<int64_t>(sizes.begin(), sizes.begin() + sparse_dim));
    const int64_t block_numel =
        product(std::vector<int64_t>(sizes.begin() + sparse_dim, sizes.end()));

    std::vector<int64_t> block_positions;
    dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        block_positions.reserve(static_cast<size_t>(outer_numel));
        for (int64_t block = 0; block < outer_numel; ++block) {
            const scalar_t* block_data = data + block * block_numel;
            bool nonzero = false;
            for (int64_t offset = 0; offset < block_numel; ++offset) {
                if (block_data[offset] != scalar_t(0)) {
                    nonzero = true;
                    break;
                }
            }
            if (nonzero) block_positions.push_back(block);
        }
    });

    const int64_t nnz = static_cast<int64_t>(block_positions.size());
    Tensor indices = Tensor::empty({sparse_dim, nnz}, DType::Int64, self.device());
    std::vector<int64_t> values_shape{nnz};
    values_shape.insert(values_shape.end(), sizes.begin() + sparse_dim, sizes.end());
    Tensor values = Tensor::empty(values_shape, contiguous_self.dtype(), self.device());
    int64_t* index_data = indices.data_ptr<int64_t>();

    dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = contiguous_self.data_ptr<scalar_t>();
        scalar_t* destination = values.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            int64_t remainder = block_positions[static_cast<size_t>(n)];
            for (int64_t d = sparse_dim - 1; d >= 0; --d) {
                const int64_t dim_size = sizes[static_cast<size_t>(d)];
                index_data[d * nnz + n] = remainder % dim_size;
                remainder /= dim_size;
            }
            std::copy_n(source + block_positions[static_cast<size_t>(n)] * block_numel,
                        block_numel, destination + n * block_numel);
        }
    });
    return Tensor::make_sparse_coo_tensor(indices, values, sizes, true);
}

} // namespace

Tensor to_sparse_coo_cpu(const Tensor& self) {
    if (self.is_sparse_csr()) return csr_to_coo_cpu(self).coalesce();
    if (self.is_sparse()) return self.coalesce();
    Tensor contiguous_self = self.contiguous();
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(contiguous_self.shape());
    const int64_t ndim = static_cast<int64_t>(sizes.size());
    const std::vector<int64_t> positions = nonzero_positions(contiguous_self);
    const int64_t nnz = static_cast<int64_t>(positions.size());

    Tensor indices = Tensor::empty({ndim, nnz}, DType::Int64, self.device());
    Tensor values = Tensor::empty(
        std::vector<int64_t>{nnz}, contiguous_self.dtype(), self.device());
    int64_t* index_data = indices.data_ptr<int64_t>();
    dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = contiguous_self.data_ptr<scalar_t>();
        scalar_t* destination = values.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            int64_t remainder = positions[static_cast<size_t>(n)];
            destination[n] = source[remainder];
            for (int64_t d = ndim - 1; d >= 0; --d) {
                index_data[d * nnz + n] = remainder % sizes[static_cast<size_t>(d)];
                remainder /= sizes[static_cast<size_t>(d)];
            }
        }
    });
    return Tensor::make_sparse_coo_tensor(indices, values, sizes, /*is_coalesced=*/true);
}

Tensor to_sparse_coo_cpu_sparse_dim(const Tensor& self, int64_t sparse_dim) {
    if (self.is_sparse_csr()) {
        if (sparse_dim != 2) {
            TP_THROW(ValueError,
                     "to_sparse(): compressed input requires sparse_dim=2");
        }
        return csr_to_coo_cpu(self).coalesce();
    }
    if (self.is_sparse()) {
        if (sparse_dim != self.sparse_dim()) {
            TP_THROW(ValueError,
                     "to_sparse(): sparse_dim must match the sparse input");
        }
        return self.coalesce();
    }
    return dense_to_sparse_coo_cpu(self, sparse_dim);
}

Tensor to_sparse_csr_cpu(const Tensor& self) {
    if (self.dim() != 2) {
        TP_THROW(RuntimeError,
                 "to_sparse_csr(): only 2-D input is supported, got " +
                     std::to_string(self.dim()) + "-D");
    }
    if (self.is_sparse_csr()) return self;
    if (self.is_sparse()) {
        return coo_to_csr_cpu(self.coalesce(), self.size(0));
    }
    Tensor contiguous_self = self.contiguous();
    const int64_t rows = contiguous_self.size(0);
    const int64_t cols = contiguous_self.size(1);
    const std::vector<int64_t> positions = nonzero_positions(contiguous_self);

    Tensor crow = Tensor::zeros({rows + 1}, DType::Int64, self.device());
    int64_t* crow_ptr = crow.data_ptr<int64_t>();
    const int64_t nnz = static_cast<int64_t>(positions.size());
    for (int64_t position : positions) {
        const int64_t row = position / cols;
        ++crow_ptr[row + 1];
    }
    for (int64_t i = 0; i < rows; ++i) crow_ptr[i + 1] += crow_ptr[i];

    Tensor col = Tensor::empty({nnz}, DType::Int64, self.device());
    Tensor values = Tensor::empty({nnz}, contiguous_self.dtype(), self.device());
    int64_t* col_ptr = col.data_ptr<int64_t>();
    dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* source = contiguous_self.data_ptr<scalar_t>();
        scalar_t* destination = values.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            const int64_t position = positions[static_cast<size_t>(n)];
            col_ptr[n] = position % cols;
            destination[n] = source[position];
        }
    });
    // Row-major scanning keeps columns ascending within every row, so the
    // result is canonical CSR.
    return Tensor::make_sparse_csr_tensor(crow, col, values, {rows, cols});
}

Tensor sparse_mm_cpu(const Tensor& self, const Tensor& dense) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError,
                 "sparse_mm(): expected a sparse COO/CSR first argument");
    }
    if (self.dim() != 2 || dense.dim() != 2) {
        TP_THROW(RuntimeError, "sparse_mm(): both operands must be 2-D");
    }
    if (self.device() != dense.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mm(): operands must be on the same device");
    }
    const int64_t inner = self.size(1);
    if (dense.size(0) != inner) {
        TP_THROW(RuntimeError,
                 "sparse_mm(): operand shapes are incompatible for matmul");
    }
    if (dense.dtype() != self.dtype()) {
        TP_THROW(TypeError,
                 "sparse_mm(): operands must share the sparse tensor's dtype");
    }
    const int64_t rows = self.size(0);
    const int64_t cols = dense.size(1);
    Tensor dense_contiguous = dense.is_contiguous() ? dense : dense.contiguous();
    Tensor out = Tensor::zeros({rows, cols}, self.dtype(), self.device());

    if (self.is_sparse_csr()) {
        if (self.sparse_dim() != 2 || self._values().dim() != 1) {
            TP_THROW(RuntimeError,
                     "sparse_mm(): hybrid CSR tensors are not supported");
        }
        Tensor crow = self._crow_indices().contiguous();
        Tensor col = self._col_indices().contiguous();
        Tensor values = self._values().contiguous();
        const int64_t* crow_ptr = crow.data_ptr<int64_t>();
        const int64_t* col_ptr = col.data_ptr<int64_t>();
        dispatch_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            const scalar_t* value_ptr = values.data_ptr<scalar_t>();
            const scalar_t* dense_ptr = dense_contiguous.data_ptr<scalar_t>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t t = crow_ptr[i]; t < crow_ptr[i + 1]; ++t) {
                    const scalar_t v = value_ptr[t];
                    const scalar_t* dense_row = dense_ptr + col_ptr[t] * cols;
                    scalar_t* out_row = out_ptr + i * cols;
                    for (int64_t j = 0; j < cols; ++j) out_row[j] += v * dense_row[j];
                }
            }
        });
        return out;
    }

    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    if (values.dim() != 1) {
        TP_THROW(RuntimeError, "sparse_mm(): hybrid COO tensors are not supported");
    }
    const int64_t nnz = indices.size(1);
    const int64_t* index_data = indices.data_ptr<int64_t>();
    dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const int64_t* row_indices = index_data;
        const int64_t* col_indices = index_data + nnz;
        const scalar_t* value_ptr = values.data_ptr<scalar_t>();
        const scalar_t* dense_ptr = dense_contiguous.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            const scalar_t v = value_ptr[n];
            const scalar_t* dense_row = dense_ptr + col_indices[n] * cols;
            scalar_t* out_row = out_ptr + row_indices[n] * cols;
            for (int64_t j = 0; j < cols; ++j) out_row[j] += v * dense_row[j];
        }
    });
    return out;
}

// returns a dense tensor (the values summed); a partial reduction keeps the
// surviving coordinate rows, rebuilds the COO over the kept dims and folds
// duplicate coordinates via coalesce(), returning a sparse tensor.  ``dtype``
// converts the input first, acting as the accumulation type.
Tensor sparse_sum_cpu(const Tensor& self, std::optional<std::vector<int64_t>> dim,
                      std::optional<DType> dtype) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_sum(): expected a sparse tensor");
    }
    Tensor input = self;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != self.dtype()) {
        input = self.to(*dtype);
    }
    const bool reduce_all = !dim.has_value() || dim->empty();
    Tensor canonical;
    if (input.is_sparse_csr()) {
        canonical = reduce_all ? input : csr_to_coo_cpu(input).coalesce();
    } else {
        canonical = input.is_coalesced() ? input : input.coalesce();
    }
    if (canonical._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse_sum(): hybrid COO tensors are not supported");
    }

    // No dims (or an empty list): dense sum over all values.
    if (reduce_all) {
        return canonical._values().sum();
    }

    const int64_t sparse_dim = canonical.sparse_dim();
    std::vector<bool> reduced(static_cast<size_t>(sparse_dim), false);
    for (int64_t d : *dim) {
        if (d < 0) d += canonical.dim();
        if (d < 0 || d >= sparse_dim) {
            TP_THROW(ValueError, "sparse_sum(): dim out of the sparse range");
        }
        reduced[static_cast<size_t>(d)] = true;
    }
    int64_t num_reduced = 0;
    for (bool r : reduced) num_reduced += r ? 1 : 0;
    if (num_reduced == sparse_dim) {
        return canonical._values().sum();
    }

    std::vector<int64_t> kept_dims;
    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (!reduced[static_cast<size_t>(d)]) kept_dims.push_back(d);
    }
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(canonical.shape());
    std::vector<int64_t> out_sizes;
    for (int64_t d : kept_dims) out_sizes.push_back(sizes[static_cast<size_t>(d)]);

    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    const int64_t nnz = indices.size(1);
    Tensor new_indices = Tensor::empty(
        {static_cast<int64_t>(kept_dims.size()), nnz}, DType::Int64,
        indices.device());
    int64_t* dst = new_indices.data_ptr<int64_t>();
    const int64_t* src = indices.data_ptr<int64_t>();
    for (int64_t i = 0; i < static_cast<int64_t>(kept_dims.size()); ++i) {
        std::copy_n(src + kept_dims[static_cast<size_t>(i)] * nnz, nnz,
                    dst + i * nnz);
    }
    return Tensor::make_sparse_coo_tensor(new_indices, values.clone(), out_sizes,
                                          /*is_coalesced=*/false).coalesce();
}

namespace {

// Coordinate-union addition: concatenate both COO component sets and run the
// existing coalescing sweep so duplicates fold naturally.
Tensor sparse_add_cpu_impl(const Tensor& self, const Tensor& other) {
    if (!self.is_sparse() || self.is_sparse_csr() ||
        !other.is_sparse() || other.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): expected two sparse COO tensors");
    }
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): operands must have identical sizes");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(TypeError, "sparse.add(): operands must share one dtype");
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse.add(): operands must share one device");
    }
    Tensor a = self.is_coalesced() ? self : self.coalesce();
    Tensor b = other.is_coalesced() ? other : other.coalesce();
    const auto a_value_shape = a._values().shape();
    const auto b_value_shape = b._values().shape();
    if (a.sparse_dim() != b.sparse_dim() ||
        a_value_shape.size() != b_value_shape.size()) {
        TP_THROW(RuntimeError,
                 "sparse.add(): sparse dimensions and value shapes must match");
    }
    for (size_t dim = 1; dim < a_value_shape.size(); ++dim) {
        if (a_value_shape[dim] != b_value_shape[dim]) {
            TP_THROW(
                RuntimeError,
                "sparse.add(): sparse dimensions and value shapes must match");
        }
    }
    Tensor cat_indices = Tensor::cat({a._indices(), b._indices()}, 1);
    Tensor cat_values = Tensor::cat({a._values(), b._values()}, 0);
    return Tensor::make_sparse_coo_tensor(
        cat_indices, cat_values,
        static_cast<std::vector<int64_t>>(a.shape()),
        /*is_coalesced=*/false).coalesce();
}

} // namespace

Tensor sparse_add_cpu(const Tensor& self, const Tensor& other) {
    return sparse_add_cpu_impl(self, other);
}

Tensor sparse_mul_cpu(const Tensor& self, const Tensor& other) {
    Tensor result_storage;
    if (!self.is_sparse() || self.is_sparse_csr() ||
        !other.is_sparse() || other.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): expected two sparse COO tensors");
    }
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): operands must have identical sizes");
    }
    if (self.dtype() != other.dtype()) {
        TP_THROW(TypeError, "sparse.mul(): operands must share one dtype");
    }
    if (self.device() != other.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse.mul(): operands must share one device");
    }
    Tensor a = self.is_coalesced() ? self : self.coalesce();
    Tensor b = other.is_coalesced() ? other : other.coalesce();
    if (a._values().dim() != 1 || b._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse.mul(): hybrid COO tensors are not supported");
    }
    const int64_t sparse_dim = a.sparse_dim();
    Tensor ia = a._indices().contiguous();
    Tensor va = a._values().contiguous();
    Tensor ib = b._indices().contiguous();
    Tensor vb = b._values().contiguous();
    const int64_t nnz_a = va.size(0);
    const int64_t nnz_b = vb.size(0);

    // Sorted-merge intersection on coordinates.  Both sides are coalesced so
    // their rows sort lexicographically already.
    auto coord_at = [](const Tensor& idx, int64_t nnz, int64_t n,
                       std::vector<int64_t>* out) {
        const int64_t* data = idx.data_ptr<int64_t>();
        for (int64_t d = 0; d < idx.size(0); ++d) {
            out->at(static_cast<size_t>(d)) = data[d * nnz + n];
        }
    };

    dispatch_dtype(va.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        std::vector<int64_t> out_coords;
        std::vector<scalar_t> out_values;
        std::vector<int64_t> ca(static_cast<size_t>(sparse_dim));
        std::vector<int64_t> cb(static_cast<size_t>(sparse_dim));
        const scalar_t* pa = va.data_ptr<scalar_t>();
        const scalar_t* pb = vb.data_ptr<scalar_t>();
        int64_t i = 0, j = 0;
        while (i < nnz_a && j < nnz_b) {
            coord_at(ia, nnz_a, i, &ca);
            coord_at(ib, nnz_b, j, &cb);
            if (ca < cb) { ++i; }
            else if (cb < ca) { ++j; }
            else {
                out_coords.insert(out_coords.end(), ca.begin(), ca.end());
                out_values.push_back(pa[i] * pb[j]);
                ++i; ++j;
            }
        }
        const int64_t out_nnz = static_cast<int64_t>(out_values.size());
        Tensor oi = Tensor::empty({sparse_dim, out_nnz}, DType::Int64,
                                  self.device());
        Tensor ov = Tensor::empty(std::vector<int64_t>{out_nnz},
                                  self.dtype(), self.device());
        int64_t* oi_ptr = oi.data_ptr<int64_t>();
        for (int64_t n = 0; n < out_nnz; ++n) {
            for (int64_t d = 0; d < sparse_dim; ++d) {
                oi_ptr[d * out_nnz + n] =
                    out_coords[static_cast<size_t>(n * sparse_dim + d)];
            }
        }
        for (int64_t n = 0; n < out_nnz; ++n) {
            ov.data_ptr<scalar_t>()[n] = out_values[static_cast<size_t>(n)];
        }
        result_storage = Tensor::make_sparse_coo_tensor(
            oi, ov, static_cast<std::vector<int64_t>>(a.shape()), true);
    });
    return result_storage;
}

// stores ``min(d+M, L)`` entries when ``d <= 0`` and ``min(N, L) - d``
// otherwise, placed starting at cell ``(max(d,0)-d, max(d,0))``; values are
// read from row ``j`` of ``diagonals`` beginning at column ``max(d, 0)``.
namespace {

// Builds canonical CSR components from a coalesced COO whose coordinates are
// sorted row-major over a 2-D shape.
Tensor coo_to_csr_cpu(const Tensor& coalesced, int64_t rows) {
    Tensor indices = coalesced._indices().contiguous();
    Tensor values = coalesced._values().contiguous();
    const int64_t nnz = indices.size(1);
    const int64_t* coords = indices.data_ptr<int64_t>();
    Tensor crow = Tensor::zeros({rows + 1}, DType::Int64, coalesced.device());
    int64_t* crow_ptr = crow.data_ptr<int64_t>();
    for (int64_t n = 0; n < nnz; ++n) ++crow_ptr[coords[n] + 1];
    for (int64_t i = 0; i < rows; ++i) crow_ptr[i + 1] += crow_ptr[i];
    return Tensor::make_sparse_csr_tensor(
        crow, indices.select(0, 1), values,
        static_cast<std::vector<int64_t>>(coalesced.shape()));
}

} // namespace

Tensor spdiags_cpu(const Tensor& diagonals, const Tensor& offsets,
                   std::vector<int64_t> shape,
                   std::optional<int64_t> layout) {
    if (layout.has_value() && *layout != 0 && *layout != 1) {
        TP_THROW(ValueError,
                 "spdiags(): only sparse_coo (0) and sparse_csr (1) output "
                 "layouts are supported");
    }
    if (shape.size() != 2) {
        TP_THROW(ValueError, "spdiags(): output shape must be 2-dimensional");
    }
    Tensor diags2d = diagonals.dim() == 1 ? diagonals.unsqueeze(0) : diagonals;
    if (diags2d.dim() != 2) {
        TP_THROW(ValueError, "spdiags(): diagonals must be a vector or matrix");
    }
    if (diags2d.device() != offsets.device()) {
        TP_THROW(DeviceMismatchError,
                 "spdiags(): diagonals and offsets must share one device");
    }
    Tensor offs = offsets.dim() == 0 ? offsets.unsqueeze(0) : offsets;
    if (offs.dim() != 1 || offs.dtype() != DType::Int64) {
        TP_THROW(TypeError, "spdiags(): offset tensor must be 1-D int64");
    }
    const int64_t n_diag = offs.size(0);
    if (diags2d.size(0) != n_diag) {
        TP_THROW(ValueError,
                 "spdiags(): number of diagonals (" +
                     std::to_string(diags2d.size(0)) +
                     ") does not match the number of offsets (" +
                     std::to_string(n_diag) + ")");
    }
    Tensor diags_c = diags2d.contiguous();
    Tensor offs_c = offs.contiguous();
    const int64_t* off_data = offs_c.data_ptr<int64_t>();
    std::vector<int64_t> off_host(off_data, off_data + n_diag);
    std::unordered_set<int64_t> unique_offs(off_host.begin(), off_host.end());
    if (unique_offs.size() != static_cast<size_t>(n_diag)) {
        TP_THROW(ValueError, "spdiags(): offset tensor contains duplicate values");
    }

    const int64_t m_size = shape[0];
    const int64_t n_size = shape[1];
    const int64_t length = diags_c.size(1);
    std::vector<int64_t> counts(static_cast<size_t>(n_diag), 0);
    std::vector<int64_t> starts(static_cast<size_t>(n_diag), 0);
    int64_t total_nnz = 0;
    for (int64_t j = 0; j < n_diag; ++j) {
        const int64_t d = off_host[static_cast<size_t>(j)];
        // would produce undefined content.
        const int64_t count = d <= 0 ? std::min(d + m_size, length)
                                     : std::min(n_size, length) - d;
        counts[static_cast<size_t>(j)] = std::max<int64_t>(count, 0);
        starts[static_cast<size_t>(j)] = total_nnz;
        total_nnz += counts[static_cast<size_t>(j)];
    }

    Tensor indices = Tensor::empty({2, total_nnz}, DType::Int64,
                                   offsets.device());
    Tensor values = Tensor::empty({total_nnz}, diags_c.dtype(),
                                  diags_c.device());
    int64_t* rows_ptr = indices.data_ptr<int64_t>();
    int64_t* cols_ptr = rows_ptr + total_nnz;
    dispatch_dtype(diags_c.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* diag_data = diags_c.data_ptr<scalar_t>();
        scalar_t* val_ptr = values.data_ptr<scalar_t>();
        for (int64_t j = 0; j < n_diag; ++j) {
            const int64_t count = counts[static_cast<size_t>(j)];
            if (count <= 0) continue;
            const int64_t first_col =
                std::max<int64_t>(off_host[static_cast<size_t>(j)], 0);
            const int64_t first_row = first_col - off_host[static_cast<size_t>(j)];
            const int64_t slot = starts[static_cast<size_t>(j)];
            const scalar_t* read = diag_data + j * length + first_col;
            for (int64_t i = 0; i < count; ++i) {
                rows_ptr[slot + i] = first_row + i;
                cols_ptr[slot + i] = first_col + i;
                val_ptr[slot + i] = read[i];
            }
        }
    });

    auto result = Tensor::make_sparse_coo_tensor(indices, values, shape,
                                                 /*is_coalesced=*/false);
    if (layout.has_value() && *layout == 1) {
        return coo_to_csr_cpu(result.coalesce(), m_size);
    }
    return result;
}

// sparse @ dense producing a sparse result.  For every row with at least one
// stored element the product row is emitted in full (zero entries included),
// so the output preserves the input's row structure; rows without stored
// elements are dropped.  ``beta`` scales the accumulated input (unused by
// smm, which passes an empty accumulator) and ``alpha`` scales the product.
Tensor sparse_sspaddmm_cpu(const Tensor& t, const Tensor& sparse,
                           const Tensor& dense, Scalar beta, Scalar alpha) {
    if (!sparse.is_sparse() || sparse.is_sparse_csr()) {
        TP_THROW(RuntimeError,
                 "sspaddmm(): expected 'mat1' to be a sparse COO tensor");
    }
    if (sparse.sparse_dim() != 2) {
        TP_THROW(RuntimeError, "sspaddmm(): Argument #2: matrices expected, got ",
                 sparse.sparse_dim(), "D tensor");
    }
    if (sparse.dense_dim() != 0) {
        TP_THROW(RuntimeError,
                 "sspaddmm(): Argument #2: scalar values expected, got ",
                 sparse.dense_dim(), "D values");
    }
    if (dense.dim() != 2) {
        TP_THROW(RuntimeError, "sspaddmm(): Argument #3: matrices expected, got ",
                 dense.dim(), "D tensor");
    }
    TP_CHECK(dense.device() == sparse.device() && dense.device() == t.device(),
             "sspaddmm(): all operands must be on the same device");

    Tensor canonical = sparse.is_coalesced() ? sparse : sparse.coalesce();
    const int64_t dim_i = canonical.size(0);
    const int64_t dim_j = canonical.size(1);
    const int64_t dim_k = dense.size(1);

    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    if (indices.dtype() != DType::Int64) {
        indices = indices.to(DType::Int64);
    }

    const int64_t nnz = canonical._nnz();
    // Row boundaries of the coalesced matrix in CSR form.
    Tensor crow = Tensor::zeros({dim_i + 1}, DType::Int64, sparse.device());
    int64_t* crow_ptr = crow.data_ptr<int64_t>();
    const int64_t* index_data = indices.data_ptr<int64_t>();
    for (int64_t n = 0; n < nnz; ++n) {
        ++crow_ptr[index_data[n] + 1];
    }
    for (int64_t i = 0; i < dim_i; ++i) crow_ptr[i + 1] += crow_ptr[i];

    const int64_t t_nnz = t.is_sparse() ? t._nnz() : 0;
    const int64_t r_nnz = nnz * dim_k + t_nnz;
    Tensor newi = Tensor::empty({2, r_nnz}, DType::Int64, sparse.device());
    Tensor newv = Tensor::zeros({r_nnz}, values.dtype(), sparse.device());

    int64_t* newi_ptr = newi.data_ptr<int64_t>();
    Tensor dense_c = dense.contiguous();

    // Accumulated input term: beta * t, stored first.  An empty
    // accumulator (the smm spelling) contributes nothing.
    int64_t p = 0;
    if (t_nnz != 0) {
        TP_CHECK(t.dim() == 2, "sspaddmm(): Argument #1: matrices expected");
        TP_CHECK(t.size(0) == dim_i && t.size(1) == dim_k,
                 "sspaddmm(): accumulator shape mismatch");
        Tensor t_coalesced = t.is_coalesced() ? t : t.coalesce();
        Tensor t_indices = t_coalesced._indices().contiguous();
        Tensor t_values = t_coalesced._values().contiguous();
        if (t_indices.dtype() != DType::Int64) {
            t_indices = t_indices.to(DType::Int64);
        }
        std::copy_n(t_indices.data_ptr<int64_t>(), t_nnz, newi_ptr);
        std::copy_n(t_indices.data_ptr<int64_t>() + t_nnz, t_nnz,
                    newi_ptr + r_nnz);
        const int64_t t_dense_numel = t_coalesced._values().dim() > 1
            ? 1 : 1;
        (void)t_dense_numel;
        dispatch_dtype(values.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            scalar_t* dst = newv.data_ptr<scalar_t>();
            const scalar_t* src = t_values.data_ptr<scalar_t>();
            const scalar_t beta_value = beta.to<scalar_t>();
            for (int64_t n = 0; n < t_nnz; ++n) dst[n] = beta_value * src[n];
        });
        p = t_nnz;
    }

    // Product term: for each non-empty row, alpha * (values @ dense) row.
    dispatch_dtype(values.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* value_ptr = values.data_ptr<scalar_t>();
        const scalar_t* dense_ptr = dense_c.data_ptr<scalar_t>();
        scalar_t* newv_ptr = newv.data_ptr<scalar_t>();
        const scalar_t alpha_value = alpha.to<scalar_t>();
        const int64_t dense_stride0 = dense_c.stride(0);
        const int64_t dense_stride1 = dense_c.stride(1);

        for (int64_t h = 0; h < dim_i; ++h) {
            const int64_t row_start = crow_ptr[h];
            const int64_t row_end = crow_ptr[h + 1];
            if (row_start == row_end) continue;
            for (int64_t k = 0; k < dim_k; ++k) {
                newi_ptr[p + k] = h;
                newi_ptr[r_nnz + p + k] = k;
            }
            for (int64_t i = row_start; i < row_end; ++i) {
                const scalar_t val = value_ptr[i];
                const int64_t col = index_data[nnz + i];
                if (col < 0 || col >= dim_j) {
                    TP_THROW(IndexError,
                             "index out of bound. sspmm: ", col,
                             " not between 0 and ", dim_j);
                }
                const scalar_t scale = alpha_value * val;
                const scalar_t* dense_row = dense_ptr + col * dense_stride0;
                scalar_t* out_row = newv_ptr + p;
                for (int64_t k = 0; k < dim_k; ++k) {
                    out_row[k] += scale * dense_row[k * dense_stride1];
                }
            }
            p += dim_k;
        }
    });

    return Tensor::make_sparse_coo_tensor(
        newi.narrow(1, 0, p), newv.narrow(0, 0, p), {dim_i, dim_k}, false);
}

Tensor smm_cpu(const Tensor& self, const Tensor& mat2) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError,
                 "smm(): expected the first argument to be a sparse tensor");
    }
    Tensor result = Tensor::empty({0}, self.dtype(), self.device());
    return sparse_sspaddmm_cpu(result, self, mat2, Scalar(0.0), Scalar(1.0));
}

// ---------------------------------------------------------------------------
// Dense -> compressed sparse (CSR/CSC/BSR/BSC)
//
// The N-D input decomposes into (*batch, row, col, *dense_dims).  For the
// blocked layouts the row/col axes are additionally tiled into blocks and a
// block is stored when any element inside it is nonzero.  Batched inputs are
// joined along the compressed axis (rows for CSR/BSR, columns for CSC/BSC),
// which requires every batch to hold the same number of stored units, and
// the resulting components are unflattened back to the batch shape.
// ---------------------------------------------------------------------------

namespace {

} // namespace

Tensor to_sparse_compressed_cpu(const Tensor& self, int layout,
                                std::array<int64_t, 2> blocksize,
                                std::optional<int64_t> dense_dim_opt,
                                const char* name) {
    if (layout != TensorImpl::kSparseCSRLayout &&
        layout != TensorImpl::kSparseCSCLayout &&
        layout != TensorImpl::kSparseBSRLayout &&
        layout != TensorImpl::kSparseBSCLayout) {
        TP_THROW(ValueError, name, ": invalid compressed layout ", layout);
    }
    if (!self.device().is_cpu()) {
        TP_THROW(NotImplementedError, name,
                 ": only CPU inputs are supported");
    }
    if (self.dim() < 2) {
        TP_THROW(RuntimeError, name,
                 ": input must have at least 2 dimensions, got ", self.dim());
    }
    const int64_t dense_dim = dense_dim_opt.value_or(0);
    if (dense_dim < 0 || self.dim() - 2 - dense_dim < 0) {
        TP_THROW(ValueError, name,
                 ": dense_dim must satisfy 0 <= dense_dim <= self.dim()-2");
    }
    const bool row_compressed =
        layout == TensorImpl::kSparseCSRLayout ||
        layout == TensorImpl::kSparseBSRLayout;
    const bool blocked =
        layout == TensorImpl::kSparseBSRLayout ||
        layout == TensorImpl::kSparseBSCLayout;
    if (blocked) {
        if (blocksize[0] <= 0 || blocksize[1] <= 0) {
            TP_THROW(ValueError, name, ": block sizes must be positive");
        }
    } else {
        blocksize = {1, 1};
    }
    const int64_t b0 = blocksize[0];
    const int64_t b1 = blocksize[1];

    // Shape bookkeeping: (*batch, r, c, *dense).
    const int64_t ndim = self.dim();
    const int64_t n_batch_dim = ndim - 2 - dense_dim;
    const int64_t rows = self.size(n_batch_dim);
    const int64_t cols = self.size(n_batch_dim + 1);
    if (blocked) {
        if (rows % b0 != 0 || cols % b1 != 0) {
            TP_THROW(RuntimeError, name,
                     ": sizes must be divisible by the block sizes");
        }
    }
    int64_t batch_count = 1;
    std::vector<int64_t> batch_shape;
    for (int64_t d = 0; d < n_batch_dim; ++d) {
        batch_shape.push_back(self.size(d));
        batch_count *= self.size(d);
    }
    if (batch_count == 0) {
        TP_THROW(RuntimeError, name,
                 ": Expected product of batch dimensions to be non-zero.");
    }
    int64_t dense_count = 1;
    std::vector<int64_t> dense_shape;
    for (int64_t d = n_batch_dim + 2; d < ndim; ++d) {
        dense_shape.push_back(self.size(d));
        dense_count *= self.size(d);
    }

    Tensor dense = self.contiguous();

    // Combined-matrix geometry: rows_m x cols_n.  Batches are joined along
    // the compressed axis so the conversion runs over one matrix.
    const int64_t rows_m = row_compressed ? batch_count * rows : rows;
    const int64_t cols_n = row_compressed ? cols : batch_count * cols;
    // Compressed/plain units (1 per element when unblocked).
    const int64_t comp_extent = row_compressed ? (blocked ? b0 : 1)
                                               : (blocked ? b1 : 1);
    const int64_t plain_extent = row_compressed ? (blocked ? b1 : 1)
                                                : (blocked ? b0 : 1);
    const int64_t comp_units = (row_compressed ? rows_m : cols_n) / comp_extent;
    const int64_t plain_units = (row_compressed ? cols_n : rows_m) / plain_extent;

    std::vector<int64_t> compressed_vec(comp_units + 1, 0);
    // Plain coordinates and the per-unit value payloads are gathered in the
    // second pass below; count first, then prefix-sum.
    auto element_at = [&](int64_t m, int64_t n, int64_t d) -> int64_t {
        // Map combined-matrix coordinates back to (batch, row, col).
        int64_t batch, row, col;
        if (row_compressed) {
            batch = m / rows;
            row = m % rows;
            col = n;
        } else {
            batch = n / cols;
            col = n % cols;
            row = m;
        }
        return (((batch * rows + row) * cols + col) * dense_count) + d;
    };
    // Map (compressed unit, plain unit, intra-unit offsets) to combined
    // matrix coordinates.  The compressed axis enumerates rows for CSR/BSR
    // and columns for CSC/BSC; the plain axis holds the other coordinate.
    auto unit_coords = [&](int64_t cu, int64_t pu, int64_t i, int64_t j,
                           int64_t& m, int64_t& n) {
        if (row_compressed) {
            m = cu * comp_extent + i;
            n = pu * plain_extent + j;
        } else {
            n = cu * comp_extent + i;
            m = pu * plain_extent + j;
        }
    };
    auto any_nonzero = [&](int64_t cu, int64_t pu) {
        bool found = false;
        dispatch_dtype(dense.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            const scalar_t* data = dense.data_ptr<scalar_t>();
            for (int64_t i = 0; i < comp_extent && !found; ++i) {
                for (int64_t j = 0; j < plain_extent && !found; ++j) {
                    int64_t m, n;
                    unit_coords(cu, pu, i, j, m, n);
                    for (int64_t d = 0; d < dense_count; ++d) {
                        if (data[element_at(m, n, d)] != scalar_t(0)) {
                            found = true;
                            break;
                        }
                    }
                }
            }
        });
        return found;
    };

    for (int64_t cu = 0; cu < comp_units; ++cu) {
        int64_t count = 0;
        for (int64_t pu = 0; pu < plain_units; ++pu) {
            if (any_nonzero(cu, pu)) ++count;
        }
        compressed_vec[static_cast<size_t>(cu) + 1] = count;
    }
    for (int64_t cu = 0; cu < comp_units; ++cu) {
        compressed_vec[static_cast<size_t>(cu) + 1] +=
            compressed_vec[static_cast<size_t>(cu)];
    }
    const int64_t nnz = compressed_vec[static_cast<size_t>(comp_units)];

    // Every batch must hold the same number of stored units for the
    // unflattening to be well-defined.
    {
        const int64_t units_per_batch = comp_units / batch_count;
        for (int64_t k = 1; k < batch_count; ++k) {
            const int64_t prev =
                compressed_vec[static_cast<size_t>(k * units_per_batch)] -
                compressed_vec[static_cast<size_t>((k - 1) * units_per_batch)];
            const int64_t curr =
                compressed_vec[static_cast<size_t>((k + 1) * units_per_batch)] -
                compressed_vec[static_cast<size_t>(k * units_per_batch)];
            if (prev != curr) {
                TP_THROW(RuntimeError, name,
                         ": Expect the same number of specified elements per batch.");
            }
        }
    }

    Tensor compressed = Tensor::empty(
        {comp_units + 1}, DType::Int64, self.device());
    std::copy_n(compressed_vec.begin(), comp_units + 1,
                compressed.data_ptr<int64_t>());
    Tensor plain = Tensor::empty({nnz}, DType::Int64, self.device());

    // Values payload per stored unit: (comp_extent, plain_extent, *dense).
    std::vector<int64_t> values_shape;
    values_shape.push_back(nnz);
    if (blocked) {
        values_shape.push_back(b0);  // row block extent
        values_shape.push_back(b1);  // col block extent
    }
    values_shape.insert(values_shape.end(), dense_shape.begin(), dense_shape.end());
    Tensor values = Tensor::empty(values_shape, dense.dtype(), self.device());

    dispatch_dtype(dense.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* data = dense.data_ptr<scalar_t>();
        scalar_t* values_ptr = values.numel() > 0
            ? values.data_ptr<scalar_t>() : nullptr;
        int64_t* plain_ptr = plain.numel() > 0
            ? plain.data_ptr<int64_t>() : nullptr;
        int64_t slot = 0;
        for (int64_t cu = 0; cu < comp_units; ++cu) {
            for (int64_t pu = 0; pu < plain_units; ++pu) {
                if (!any_nonzero(cu, pu)) continue;
                if (plain_ptr != nullptr) plain_ptr[slot] = pu;
                for (int64_t i = 0; i < comp_extent; ++i) {
                    for (int64_t j = 0; j < plain_extent; ++j) {
                        int64_t m, n;
                        unit_coords(cu, pu, i, j, m, n);
                        // Values layout: the two trailing block dims are
                        // always (row, col) regardless of which axis is
                        // compressed.
                        const int64_t row_off =
                            m % (blocked ? b0 : 1);
                        const int64_t col_off =
                            n % (blocked ? b1 : 1);
                        for (int64_t d = 0; d < dense_count; ++d) {
                            if (values_ptr != nullptr) {
                                values_ptr
                                    [(slot * b0 + row_off) * b1 * dense_count +
                                     col_off * dense_count + d] =
                                        data[element_at(m, n, d)];
                            }
                        }
                    }
                }
                ++slot;
            }
        }
    });

    // Unflatten the components back to the batch shape.
    const int64_t units_per_batch = comp_units / batch_count;
    Tensor compressed_out, plain_out, values_out;
    if (batch_count == 1) {
        compressed_out = compressed;
        plain_out = plain;
        values_out = values;
    } else {
        std::vector<int64_t> batched_comp_shape = batch_shape;
        batched_comp_shape.push_back(units_per_batch + 1);
        compressed_out = Tensor::empty(batched_comp_shape, DType::Int64,
                                       self.device());
        int64_t* dst = compressed_out.data_ptr<int64_t>();
        for (int64_t k = 0; k < batch_count; ++k) {
            const int64_t base =
                compressed_vec[static_cast<size_t>(k * units_per_batch)];
            for (int64_t j = 0; j <= units_per_batch; ++j) {
                dst[k * (units_per_batch + 1) + j] =
                    compressed_vec[static_cast<size_t>(
                        k * units_per_batch + j)] - base;
            }
        }
        std::vector<int64_t> batched_plain_shape = batch_shape;
        batched_plain_shape.push_back(-1);
        plain_out = plain.reshape(batched_plain_shape);
        values_out = values;
        if (!batch_shape.empty()) {
            // The first (nnz) axis of the payload splits evenly across the
            // batch (checked above); resolve the -1 to the concrete extent.
            std::vector<int64_t> batched_values_shape = batch_shape;
            const int64_t payload = values.numel() > 0
                ? values.size(0) / batch_count : 0;
            batched_values_shape.push_back(payload);
            for (size_t d = 1; d < values_shape.size(); ++d) {
                batched_values_shape.push_back(values_shape[d]);
            }
            values_out = values.reshape(batched_values_shape);
        }
    }

    std::vector<int64_t> out_sizes;
    for (int64_t d = 0; d < ndim; ++d) out_sizes.push_back(self.size(d));
    return Tensor::make_sparse_compressed_tensor(
        compressed_out, plain_out, values_out, out_sizes, layout, blocksize);
}

// ---------------------------------------------------------------------------
// _sparse_sum family
//
// The plain overloads reduce the values; the dim overloads fold the listed
// sparse dims away and sum any listed dense dims inside the values payload,
// using coalesce() to fold the surviving coordinates.
// ---------------------------------------------------------------------------

Tensor _sparse_sum_cpu(const Tensor& input) {
    TP_CHECK(input.is_sparse() && !input.is_sparse_compressed(),
             "_sparse_sum(): expected a sparse COO tensor");
    return input.coalesce()._values().sum();
}

Tensor _sparse_sum_dtype_cpu(const Tensor& input, DType dtype) {
    TP_CHECK(input.is_sparse() && !input.is_sparse_compressed(),
             "_sparse_sum(): expected a sparse COO tensor");
    return input.coalesce()._values().sum(dtype);
}

Tensor _sparse_sum_dim_cpu(const Tensor& input, std::vector<int64_t> dims_to_sum,
                           std::optional<DType> dtype) {
    TP_CHECK(input.is_sparse() && !input.is_sparse_compressed(),
             "_sparse_sum(): expected a sparse COO tensor");
    Tensor working = input;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != working.dtype()) {
        working = working.to(*dtype);
    }
    Tensor canonical = working.is_coalesced() ? working : working.coalesce();
    const int64_t input_dim = canonical.dim();
    for (auto& d : dims_to_sum) {
        if (d < 0) d += input_dim;
        TP_CHECK(d >= 0 && d < input_dim,
                 "_sparse_sum(): dimension out of range");
    }

    Tensor indices = canonical._indices();
    Tensor values = canonical._values();
    const auto sizes = static_cast<std::vector<int64_t>>(canonical.shape());
    const int64_t sparse_dim = canonical.sparse_dim();

    std::vector<int64_t> dims_to_keep;
    std::vector<int64_t> dense_dims_to_sum;
    int64_t sparse_dims_to_sum_size = 0;
    {
        std::vector<bool> summed(static_cast<size_t>(input_dim), false);
        for (int64_t d : dims_to_sum) summed[static_cast<size_t>(d)] = true;
        for (int64_t d = 0; d < input_dim; ++d) {
            if (summed[static_cast<size_t>(d)]) {
                if (d < sparse_dim) ++sparse_dims_to_sum_size;
                else dense_dims_to_sum.push_back(d + 1 - sparse_dim);
            } else if (d < sparse_dim) {
                dims_to_keep.push_back(d);
            }
        }
    }
    const bool sum_all_sparse_dim = sparse_dim == sparse_dims_to_sum_size;
    const bool sum_dense_dim = !dense_dims_to_sum.empty();

    Tensor new_values = sum_dense_dim
        ? values.sum(dense_dims_to_sum)
        : values.clone();

    if (sum_all_sparse_dim) {
        // Reducing every sparse dim yields a dense tensor.
        return new_values.sum(std::vector<int64_t>{0});
    }

    Tensor new_indices;
    if (dims_to_keep.empty()) {
        // Unreachable when sum_all_sparse_dim is false (some sparse dim
        // always survives), but keep the guard for a 0-sparse-dim input.
        new_indices = indices.clone();
    } else {
        new_indices = Tensor::empty(
            {static_cast<int64_t>(dims_to_keep.size()), canonical._nnz()},
            indices.dtype(), indices.device());
        for (int64_t i = 0; i < static_cast<int64_t>(dims_to_keep.size()); ++i) {
            if (dims_to_keep[static_cast<size_t>(i)] < sparse_dim) {
                new_indices.select(0, i).copy_(
                    indices.select(0, dims_to_keep[static_cast<size_t>(i)]));
            } else {
                break;
            }
        }
    }

    std::vector<int64_t> new_sizes;
    for (int64_t d : dims_to_keep) new_sizes.push_back(sizes[static_cast<size_t>(d)]);
    // The kept coordinates are not necessarily unique yet; coalescing folds
    // the duplicates raised by the dropped dims.
    return Tensor::make_sparse_coo_tensor(
        new_indices, new_values, new_sizes, false).coalesce();
}

Tensor _sparse_sum_dim_dtype_cpu(const Tensor& input,
                                 std::vector<int64_t> dims_to_sum,
                                 DType dtype) {
    return _sparse_sum_dim_cpu(input, std::move(dims_to_sum), dtype);
}

Tensor _sparse_sum_dim_cpu_2(const Tensor& input,
                             std::vector<int64_t> dims_to_sum) {
    return _sparse_sum_dim_cpu(input, std::move(dims_to_sum), std::nullopt);
}

Tensor _sparse_sum_backward_cpu(const Tensor& grad_, const Tensor& input_,
                                std::vector<int64_t> dims_to_sum) {
    TP_CHECK(input_.is_sparse() && !input_.is_sparse_compressed(),
             "_sparse_sum_backward(): expected a sparse COO tensor");
    if ((grad_.is_sparse() && grad_._nnz() == 0) || grad_.numel() == 0) {
        return Tensor::zeros(static_cast<std::vector<int64_t>>(input_.shape()),
                             grad_.dtype(), grad_.device());
    }

    Tensor input = input_.is_coalesced() ? input_ : input_.coalesce();
    const int64_t input_dim = input.dim();
    for (auto& d : dims_to_sum) {
        if (d < 0) d += input_dim;
        TP_CHECK(d >= 0 && d < input_dim,
                 "_sparse_sum_backward(): dimension out of range");
    }

    Tensor input_indices = input._indices().contiguous();
    Tensor input_values = input._values();
    const auto input_sizes =
        static_cast<std::vector<int64_t>>(input.shape());
    const int64_t input_sparse_dim = input.sparse_dim();
    const int64_t input_dense_dim = input.dense_dim();
    const int64_t input_nnz = input._nnz();

    std::vector<bool> summed(static_cast<size_t>(input_dim), false);
    for (int64_t d : dims_to_sum) summed[static_cast<size_t>(d)] = true;
    int64_t sparse_dims_to_sum_size = 0;
    std::vector<int64_t> sparse_dims_to_keep;
    std::vector<int64_t> dense_dims_to_sum;
    for (int64_t d = 0; d < input_dim; ++d) {
        if (summed[static_cast<size_t>(d)]) {
            if (d < input_sparse_dim) ++sparse_dims_to_sum_size;
            else dense_dims_to_sum.push_back(d + 1 - input_sparse_dim);
        } else if (d < input_sparse_dim) {
            sparse_dims_to_keep.push_back(d);
        }
    }

    const bool sum_all_sparse_dim = input_sparse_dim == sparse_dims_to_sum_size;
    const bool sum_dense_dim = !dense_dims_to_sum.empty();
    const bool sum_sparse_dim = sparse_dims_to_sum_size > 0;

    if (sum_all_sparse_dim) {
        TP_CHECK(!grad_.is_sparse(),
                 "_sparse_sum_backward(): expected grad to be dense since all "
                 "sparse dims are summed");
        Tensor grad_input_values = grad_;
        auto expand_size = static_cast<std::vector<int64_t>>(input_values.shape());
        if (sum_dense_dim) {
            std::vector<int64_t> dense_expand_size = expand_size;
            dense_expand_size.erase(dense_expand_size.begin());
            for (int64_t d : dense_dims_to_sum) {
                grad_input_values = grad_input_values.unsqueeze(d - 1);
            }
            grad_input_values = grad_input_values.expand(dense_expand_size);
        }
        grad_input_values = grad_input_values.expand(expand_size).contiguous();
        return Tensor::make_sparse_coo_tensor(
            input_indices.clone(), grad_input_values, input_sizes,
            input.is_coalesced());
    }

    TP_CHECK(grad_.is_sparse(),
             "_sparse_sum_backward(): expected grad to be sparse, but got dense");
    Tensor grad = grad_.is_coalesced() ? grad_ : grad_.coalesce();
    Tensor grad_indices = grad._indices().contiguous();
    Tensor grad_values = grad._values().contiguous();
    const int64_t grad_sparse_dim = grad.sparse_dim();
    const int64_t grad_nnz = grad._nnz();

    Tensor grad_values_expand = grad_values;
    if (sum_dense_dim) {
        auto expand_size =
            static_cast<std::vector<int64_t>>(input_values.shape());
        if (sum_sparse_dim) expand_size[0] = grad_values.size(0);
        for (int64_t d : dense_dims_to_sum) {
            grad_values_expand = grad_values_expand.unsqueeze(d);
        }
        grad_values_expand = grad_values_expand.expand(expand_size).contiguous();
    }

    Tensor grad_input_values;
    if (sum_sparse_dim) {
        // Scatter the gradient back: each input coordinate keeps its own slot
        // and receives the grad value with the same surviving coordinates
        // (binary search over the flattened, sorted grad coordinates).
        grad_input_values = Tensor::zeros(
            static_cast<std::vector<int64_t>>(input_values.shape()),
            grad_values.dtype(), grad_values.device());

        // Horner-style linearization of the given coordinate dims; the
        // coalesced coordinate rows are lexicographically sorted, so the
        // flattened 1-D codes come out sorted without a re-sort.
        auto flatten_by_dims = [](const Tensor& idx,
                                  const std::vector<int64_t>& sizes,
                                  const std::vector<int64_t>& dims) {
            const int64_t nnz_count = idx.size(1);
            Tensor flat = Tensor::zeros({nnz_count}, DType::Int64,
                                        idx.device());
            int64_t* flat_ptr = flat.data_ptr<int64_t>();
            const int64_t* idx_ptr = idx.data_ptr<int64_t>();
            for (int64_t d : dims) {
                for (int64_t n = 0; n < nnz_count; ++n) {
                    flat_ptr[n] = flat_ptr[n] * sizes[static_cast<size_t>(d)] +
                                  idx_ptr[d * nnz_count + n];
                }
            }
            return flat;
        };

        std::vector<int64_t> grad_keep;
        for (int64_t d = 0; d < grad_sparse_dim; ++d) grad_keep.push_back(d);
        Tensor grad_flat = flatten_by_dims(grad_indices, input_sizes, grad_keep);
        Tensor input_flat =
            flatten_by_dims(input_indices, input_sizes, sparse_dims_to_keep);
        const int64_t* grad_flat_ptr = grad_flat.data_ptr<int64_t>();
        const int64_t* input_flat_ptr = input_flat.data_ptr<int64_t>();
        dispatch_dtype(grad_values.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            scalar_t* dst = grad_input_values.data_ptr<scalar_t>();
            const scalar_t* src = grad_values_expand.data_ptr<scalar_t>();
            const int64_t row_numel = grad_input_values.numel() /
                                      (input_nnz > 0 ? input_nnz : 1);
            for (int64_t i = 0; i < input_nnz; ++i) {
                const int64_t input_idx = input_flat_ptr[i];
                int64_t l = 0, r = grad_nnz - 1;
                while (l <= r) {
                    const int64_t m = l + (r - l) / 2;
                    if (grad_flat_ptr[m] == input_idx) {
                        std::copy_n(src + m * row_numel, row_numel,
                                    dst + i * row_numel);
                        break;
                    }
                    if (grad_flat_ptr[m] < input_idx) l = m + 1;
                    else r = m - 1;
                }
            }
        });
    } else {
        grad_input_values = grad_values_expand;
    }
    return Tensor::make_sparse_coo_tensor(
        input_indices.clone(), grad_input_values, input_sizes,
        input.is_coalesced());
}

// ---------------------------------------------------------------------------
// native_norm
//
// Norm over a sparse tensor: only full reductions are supported, the input
// must not be hybrid, and the reduced values carry the dense norm.
// ---------------------------------------------------------------------------

Tensor native_norm_cpu(const Tensor& self, Scalar p) {
    return native_norm_dim_cpu(self, p, {}, false, std::nullopt);
}

Tensor native_norm_dim_cpu(const Tensor& self, std::optional<Scalar> p,
                           std::vector<int64_t> dims, bool keepdim,
                           std::optional<DType> dtype) {
    TP_CHECK(self.is_sparse(), "norm(): expected a sparse tensor");
    (void)dims;
    TP_CHECK(dims.empty() ||
                 static_cast<int64_t>(dims.size()) == self.dim(),
             "norm(): currently only supports full reductions");
    TP_CHECK(!keepdim, "norm(): currently does not support keepdim=True");
    TP_CHECK(!dtype.has_value(), "norm(): currently does not support 'dtype'");
    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    const double p_value = p.has_value() ? p->toDouble() : 2.0;
    return canonical._values().norm(p_value);
}

} // namespace cpu
} // namespace tensorplay
