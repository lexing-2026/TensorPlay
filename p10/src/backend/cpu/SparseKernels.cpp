#include "SparseKernels.h"
#include "Utils.h"

#include <algorithm>
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
                             const std::vector<int64_t>& size, bool is_coalesced) {
    return Tensor::make_sparse_coo_tensor(indices, values, size, is_coalesced);
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

} // namespace cpu
} // namespace tensorplay
