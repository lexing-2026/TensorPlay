#include "SparseKernels.h"
#include "Utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

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
    // torch infers missing sizes from the coordinates (sparse dims: max+1)
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
    // Torch projects the dense values onto the mask's existing COO entries;
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

Tensor sparse_sum_cpu(const Tensor& self) {
    if (!self.is_sparse()) {
        TP_THROW(RuntimeError, "sparse_sum(): expected a sparse tensor");
    }
    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor values = canonical._values().contiguous();
    Tensor out = Tensor::zeros({}, self.dtype(), self.device());
    const int64_t numel = values.numel();
    dispatch_dtype(values.dtype(), [&](auto tag) {
        using scalar_t = typename decltype(tag)::type;
        const scalar_t* data = values.data_ptr<scalar_t>();
        scalar_t accumulator = scalar_t(0);
        for (int64_t i = 0; i < numel; ++i) accumulator += data[i];
        out.data_ptr<scalar_t>()[0] = accumulator;
    });
    return out;
}

} // namespace cpu
} // namespace tensorplay
