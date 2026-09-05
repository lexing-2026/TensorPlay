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
    if (!mask.is_sparse()) TP_THROW(RuntimeError, "sparse_mask(): mask must be sparse COO");
    if (dense.device() != mask.device()) {
        TP_THROW(DeviceMismatchError,
                 "sparse_mask(): dense and mask must be on the same device");
    }
    if (dense.shape() != mask.shape()) {
        TP_THROW(RuntimeError,
                 "sparse_mask(): operands have incompatible sizes; self and mask must have the same shape");
    }
    // it does not coalesce an uncoalesced mask or change its duplicate/order
    // semantics.  SparseAdam passes a coalesced gradient when it needs the
    // canonical form explicitly.
    Tensor indices = mask._indices().contiguous();
    const int64_t sparse_dim = mask.sparse_dim();
    const int64_t nnz = indices.size(1);
    const std::vector<int64_t> dense_shape = dense_shape_for(mask);
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
        mask.is_coalesced());
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

Tensor to_dense_sparse_cpu(const Tensor& self) {
    if (!self.is_sparse()) return self;
    if (self.is_sparse_csr()) {
        if (self.dim() != 2) {
            TP_THROW(RuntimeError, "to_dense(): CSR tensors must be 2-D");
        }
        Tensor crow = self._crow_indices().contiguous();
        Tensor col = self._col_indices().contiguous();
        Tensor values = self._values().contiguous();
        if (values.dim() != 1) {
            TP_THROW(RuntimeError,
                     "to_dense(): hybrid CSR tensors are not supported");
        }
        const int64_t rows = crow.size(0) - 1;
        Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());
        dispatch_dtype(self.dtype(), [&](auto tag) {
            using scalar_t = typename decltype(tag)::type;
            const int64_t* crow_ptr = crow.data_ptr<int64_t>();
            const int64_t* col_ptr = col.data_ptr<int64_t>();
            const scalar_t* value_ptr = values.data_ptr<scalar_t>();
            scalar_t* out_ptr = out.data_ptr<scalar_t>();
            for (int64_t i = 0; i < rows; ++i) {
                for (int64_t t = crow_ptr[i]; t < crow_ptr[i + 1]; ++t) {
                    out_ptr[i * self.size(1) + col_ptr[t]] = value_ptr[t];
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

} // namespace

Tensor to_sparse_coo_cpu(const Tensor& self) {
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

Tensor to_sparse_csr_cpu(const Tensor& self) {
    if (self.dim() != 2) {
        TP_THROW(RuntimeError,
                 "to_sparse_csr(): only 2-D input is supported, got " +
                     std::to_string(self.dim()) + "-D");
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
    Tensor canonical = input.is_coalesced() ? input : input.coalesce();
    if (canonical._values().dim() != 1) {
        TP_THROW(RuntimeError,
                 "sparse_sum(): hybrid COO tensors are not supported");
    }

    // No dims (or an empty list): dense sum over all values.
    if (!dim.has_value() || dim->empty()) {
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

} // namespace cpu
} // namespace tensorplay
