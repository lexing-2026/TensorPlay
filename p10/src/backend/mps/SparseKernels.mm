#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "DistributionDispatch.h"
#include "OperationUtils.h"
#include "Utils.h"

#include "backend/mps/MPSDevice.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

namespace tensorplay {
namespace mps {

// Defined in FactoryKernels.mm.
Tensor& fill_kernel(Tensor& self, const Scalar& value);

// ---------------------------------------------------------------------------
// Sparse COO kernels for the MPS backend.
//
// Sparse tensors store their indices/values on the MPS device; all kernels
// below operate on the host-visible shared-memory mapping of those buffers
// (valid on unified-memory Apple silicon), mirroring the CPU semantics
// bit-for-bit.  This gives the MPS backend a complete sparse COO surface:
// factory, coalesce, conversions, arithmetic, matmul and reductions.
// Compressed layouts (CSR/CSC/BSR/BSC) are not supported yet.
// ---------------------------------------------------------------------------

namespace {

struct Entry {
    std::vector<int64_t> coordinate;
    int64_t source;
};

bool coordinate_less(const Entry& lhs, const Entry& rhs) {
    return std::lexicographical_compare(lhs.coordinate.begin(),
                                        lhs.coordinate.end(),
                                        rhs.coordinate.begin(),
                                        rhs.coordinate.end());
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
    std::vector<int64_t> shape =
        static_cast<std::vector<int64_t>>(sparse.shape());
    return std::vector<int64_t>(shape.begin() + sparse_dim, shape.end());
}

void require_sparse_coo(const Tensor& t, const char* op) {
    if (!t.is_sparse()) {
        TP_THROW(RuntimeError, std::string(op) +
             ": expected a sparse tensor");
    }
    if (t.is_sparse_compressed()) {
        TP_THROW(NotImplementedError,
                 std::string(op) + ": compressed sparse layouts are not "
                 "supported on the MPS backend yet");
    }
}

// Positions (row-major flat coordinates) of nonzero elements of a
// contiguous host tensor.
std::vector<int64_t> nonzero_positions(const Tensor& contiguous_self) {
    std::vector<int64_t> positions;
    const int64_t numel = contiguous_self.numel();
    distribution::dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* data = contiguous_self.data_ptr<scalar_t>();
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] != scalar_t(0)) positions.push_back(i);
        }
    });
    return positions;
}

} // namespace

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

Tensor sparse_coo_tensor_kernel(const Tensor& indices, const Tensor& values,
                                const std::optional<std::vector<int64_t>>& size_arg,
                                bool is_coalesced) {
    if (!indices.device().is_mps() || !values.device().is_mps()) {
        TP_THROW(RuntimeError,
                 "sparse_coo_tensor(): indices and values must be on the MPS "
                 "device");
    }
    std::optional<std::vector<int64_t>> size = size_arg;
    if (!size.has_value()) {
        const Tensor host_indices = indices.is_contiguous()
            ? indices : indices.contiguous();
        const Tensor canonical = host_indices.dtype() == DType::Int64
            ? host_indices : host_indices.to(DType::Int64);
        const int64_t sparse_dim = canonical.size(0);
        const int64_t nnz = canonical.size(1);
        std::vector<int64_t> inferred(static_cast<size_t>(sparse_dim), 0);
        const int64_t* index_data = canonical.data_ptr<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            int64_t max_coordinate = -1;
            for (int64_t n = 0; n < nnz; ++n) {
                max_coordinate = std::max(max_coordinate,
                                          index_data[d * nnz + n]);
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

// ---------------------------------------------------------------------------
// Coalesce
// ---------------------------------------------------------------------------

Tensor coalesce_kernel(const Tensor& self) {
    require_sparse_coo(self, "coalesce()");
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
                TP_THROW(IndexError,
                         "coalesce(): sparse index is out of bounds");
            }
            entry.coordinate[static_cast<size_t>(d)] = coordinate;
        }
        entries.push_back(std::move(entry));
    }
    std::stable_sort(entries.begin(), entries.end(), coordinate_less);

    int64_t output_nnz = 0;
    for (size_t i = 0; i < entries.size(); ++i) {
        if (i == 0 || !coordinate_equal(entries[i], entries[i - 1])) {
            ++output_nnz;
        }
    }

    Tensor output_indices = Tensor::empty({sparse_dim, output_nnz},
                                          DType::Int64, self.device());
    std::vector<int64_t> output_values_shape;
    output_values_shape.push_back(output_nnz);
    output_values_shape.insert(output_values_shape.end(), dense_shape.begin(),
                               dense_shape.end());
    Tensor output_values = Tensor::zeros(output_values_shape, self.dtype(),
                                         self.device());
    int64_t* output_index_data = output_indices.data_ptr<int64_t>();

    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const scalar_t* source = values.data_ptr<scalar_t>();
        scalar_t* destination = output_values.data_ptr<scalar_t>();
        int64_t output_index = -1;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (i == 0 || !coordinate_equal(entries[i], entries[i - 1])) {
                ++output_index;
                for (int64_t d = 0; d < sparse_dim; ++d) {
                    output_index_data[d * output_nnz + output_index] =
                        entries[i].coordinate[static_cast<size_t>(d)];
                }
            }
            const scalar_t* source_row =
                source + entries[i].source * dense_numel;
            scalar_t* destination_row =
                destination + output_index * dense_numel;
            for (int64_t j = 0; j < dense_numel; ++j) {
                destination_row[j] += source_row[j];
            }
        }
    });

    return Tensor::make_sparse_coo_tensor(
        output_indices, output_values,
        static_cast<std::vector<int64_t>>(self.shape()), true);
}

// ---------------------------------------------------------------------------
// to_dense (COO)
// ---------------------------------------------------------------------------

Tensor to_dense_kernel(const Tensor& self) {
    if (!self.is_sparse()) return self;
    require_sparse_coo(self, "to_dense()");
    // Coalesce first so each coordinate is written exactly once.
    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    Tensor out = Tensor::zeros(self.shape(), self.dtype(), self.device());
    const int64_t sparse_dim = canonical.sparse_dim();
    const int64_t nnz = indices.size(1);
    std::vector<int64_t> dense_shape = dense_shape_for(canonical);
    const int64_t dense_numel = product(dense_shape);
    const std::vector<int64_t> out_strides = out.strides();

    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
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
                    const int64_t dim_size =
                        dense_shape[static_cast<size_t>(d)];
                    const int64_t coordinate = remainder % dim_size;
                    remainder /= dim_size;
                    offset += coordinate *
                        out_strides[static_cast<size_t>(sparse_dim + d)];
                }
                destination[offset] += source[n * dense_numel + j];
            }
        }
    });
    return out;
}

// ---------------------------------------------------------------------------
// to_sparse (dense -> COO)
// ---------------------------------------------------------------------------

Tensor dense_to_sparse_coo(const Tensor& self, int64_t sparse_dim) {
    const int64_t ndim = self.dim();
    if (sparse_dim < 0 || sparse_dim > ndim) {
        TP_THROW(ValueError, "to_sparse(): sparse_dim must be in [0," +
             std::to_string(ndim) + "]");
    }
    if (ndim > 0 && sparse_dim == 0) {
        TP_THROW(ValueError,
                 "to_sparse(): sparse_dim must be greater than zero for a "
                 "non-scalar tensor");
    }

    Tensor contiguous_self = self.contiguous();
    const std::vector<int64_t> sizes =
        static_cast<std::vector<int64_t>>(contiguous_self.shape());
    const int64_t outer_numel =
        product(std::vector<int64_t>(sizes.begin(), sizes.begin() + sparse_dim));
    const int64_t block_numel =
        product(std::vector<int64_t>(sizes.begin() + sparse_dim, sizes.end()));

    std::vector<int64_t> block_positions;
    distribution::dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
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
    Tensor indices = Tensor::empty({sparse_dim, nnz}, DType::Int64,
                                    self.device());
    std::vector<int64_t> values_shape{nnz};
    values_shape.insert(values_shape.end(), sizes.begin() + sparse_dim,
                       sizes.end());
    Tensor values = Tensor::empty(values_shape, contiguous_self.dtype(),
                                  self.device());
    int64_t* index_data = indices.data_ptr<int64_t>();

    distribution::dispatch_dtype(contiguous_self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
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
    return Tensor::make_sparse_coo_tensor(indices, values, sizes,
                                          /*is_coalesced=*/true);
}

Tensor to_sparse_kernel(const Tensor& self) {
    if (self.is_sparse()) return self.coalesce();
    return dense_to_sparse_coo(self, self.dim());
}

Tensor to_sparse_sparse_dim_kernel(const Tensor& self, int64_t sparse_dim) {
    if (self.is_sparse()) {
        if (sparse_dim != self.sparse_dim()) {
            TP_THROW(ValueError,
                     "to_sparse(): sparse_dim must match the sparse input");
        }
        return self.coalesce();
    }
    return dense_to_sparse_coo(self, sparse_dim);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int64_t nnz_kernel(const Tensor& self) {
    require_sparse_coo(self, "_nnz()");
    return self._values().size(0);
}

// ---------------------------------------------------------------------------
// sparse_mm (COO @ dense)
// ---------------------------------------------------------------------------

Tensor sparse_mm_kernel(const Tensor& self, const Tensor& dense) {
    require_sparse_coo(self, "sparse_mm()");
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

    Tensor canonical = self.is_coalesced() ? self : self.coalesce();
    Tensor indices = canonical._indices().contiguous();
    Tensor values = canonical._values().contiguous();
    if (values.dim() != 1) {
        TP_THROW(RuntimeError, "sparse_mm(): hybrid COO tensors are not supported");
    }
    const int64_t nnz = indices.size(1);
    const int64_t* index_data = indices.data_ptr<int64_t>();

    distribution::dispatch_dtype(self.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
        const int64_t* row_indices = index_data;
        const int64_t* col_indices = index_data + nnz;
        const scalar_t* value_ptr = values.data_ptr<scalar_t>();
        const scalar_t* dense_ptr = dense_contiguous.data_ptr<scalar_t>();
        scalar_t* out_ptr = out.data_ptr<scalar_t>();
        for (int64_t n = 0; n < nnz; ++n) {
            const scalar_t v = value_ptr[n];
            const scalar_t* dense_row = dense_ptr + col_indices[n] * cols;
            scalar_t* out_row = out_ptr + row_indices[n] * cols;
            for (int64_t j = 0; j < cols; ++j) {
                out_row[j] += v * dense_row[j];
            }
        }
    });
    return out;
}

// ---------------------------------------------------------------------------
// sparse_add / sparse_mul (COO)
// ---------------------------------------------------------------------------

Tensor sparse_add_kernel(const Tensor& self, const Tensor& other) {
    require_sparse_coo(self, "sparse.add()");
    require_sparse_coo(other, "sparse.add()");
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError, "sparse.add(): operands must have identical sizes");
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
            TP_THROW(RuntimeError,
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

Tensor sparse_mul_kernel(const Tensor& self, const Tensor& other) {
    require_sparse_coo(self, "sparse.mul()");
    require_sparse_coo(other, "sparse.mul()");
    if (self.shape() != other.shape()) {
        TP_THROW(RuntimeError, "sparse.mul(): operands must have identical sizes");
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
        TP_THROW(RuntimeError, "sparse.mul(): hybrid COO tensors are not supported");
    }
    const int64_t sparse_dim = a.sparse_dim();
    Tensor ia = a._indices().contiguous();
    Tensor va = a._values().contiguous();
    Tensor ib = b._indices().contiguous();
    Tensor vb = b._values().contiguous();
    const int64_t nnz_a = va.size(0);
    const int64_t nnz_b = vb.size(0);

    auto coord_at = [](const Tensor& idx, int64_t nnz, int64_t n,
                       std::vector<int64_t>* out) {
        const int64_t* data = idx.data_ptr<int64_t>();
        for (int64_t d = 0; d < idx.size(0); ++d) {
            out->at(static_cast<size_t>(d)) = data[d * nnz + n];
        }
    };

    Tensor result_storage;
    distribution::dispatch_dtype(va.dtype(), [&](auto tag) {
        using scalar_t = decltype(tag);
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
            if (ca < cb) {
                ++i;
            } else if (cb < ca) {
                ++j;
            } else {
                out_coords.insert(out_coords.end(), ca.begin(), ca.end());
                out_values.push_back(pa[i] * pb[j]);
                ++i;
                ++j;
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
            oi, ov, static_cast<std::vector<int64_t>>(self.shape()), true);
    });
    return result_storage;
}

// ---------------------------------------------------------------------------
// sparse_sum / _sparse_sum (COO)
// ---------------------------------------------------------------------------

Tensor sparse_sum_kernel(const Tensor& self,
                         const std::optional<std::vector<int64_t>>& dim,
                         std::optional<DType> dtype) {
    require_sparse_coo(self, "sparse_sum()");
    Tensor input = self;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != self.dtype()) {
        input = self.to(*dtype);
    }
    const bool reduce_all = !dim.has_value() || dim->empty();
    Tensor canonical = input.is_coalesced() ? input : input.coalesce();
    if (canonical._values().dim() != 1) {
        TP_THROW(RuntimeError, "sparse_sum(): hybrid COO tensors are not supported");
    }

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
    for (int64_t d : kept_dims) {
        out_sizes.push_back(sizes[static_cast<size_t>(d)]);
    }

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

Tensor _sparse_sum_kernel(const Tensor& input) {
    require_sparse_coo(input, "_sparse_sum()");
    return input.coalesce()._values().sum();
}

Tensor _sparse_sum_dtype_kernel(const Tensor& input, DType dtype) {
    require_sparse_coo(input, "_sparse_sum()");
    return input.coalesce()._values().sum(dtype);
}

Tensor _sparse_sum_dim_kernel(const Tensor& input,
                              const std::vector<int64_t>& dims_to_sum,
                              std::optional<DType> dtype) {
    require_sparse_coo(input, "_sparse_sum()");
    Tensor working = input;
    if (dtype.has_value() && *dtype != DType::Undefined &&
        *dtype != working.dtype()) {
        working = working.to(*dtype);
    }
    Tensor canonical = working.is_coalesced() ? working : working.coalesce();
    const int64_t input_dim = canonical.dim();
    std::vector<int64_t> dims = dims_to_sum;
    for (auto& d : dims) {
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
        for (int64_t d : dims) summed[static_cast<size_t>(d)] = true;
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
        return new_values.sum(std::vector<int64_t>{0});
    }

    Tensor new_indices = indices.clone();
    if (!dims_to_keep.empty()) {
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
    for (int64_t d : dims_to_keep) {
        new_sizes.push_back(sizes[static_cast<size_t>(d)]);
    }
    return Tensor::make_sparse_coo_tensor(
        new_indices, new_values, new_sizes, false).coalesce();
}

Tensor _sparse_sum_dim_dtype_kernel(const Tensor& input,
                                    const std::vector<int64_t>& dims_to_sum,
                                    DType dtype) {
    return _sparse_sum_dim_kernel(input, dims_to_sum, dtype);
}

// ---------------------------------------------------------------------------
// _sparse_addmm: beta * self + alpha * (sparse @ dense)
// ---------------------------------------------------------------------------

Tensor sparse_addmm_kernel(const Tensor& self, const Tensor& mat1,
                           const Tensor& mat2, const Scalar& beta,
                           const Scalar& alpha) {
    Tensor prod = sparse_mm_kernel(mat1, mat2);
    const DType dt = prod.dtype();
    const Device& dev = prod.device();

    Tensor beta_t = Tensor({1}, dt, dev);
    fill_kernel(beta_t, beta);
    Tensor alpha_t = Tensor({1}, dt, dev);
    fill_kernel(alpha_t, alpha);

    MPSBinaryOpBlock mulBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                 MPSGraphTensor* b) {
        return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    MPSBinaryOpBlock addBlock = ^MPSGraphTensor*(MPSGraph* g, MPSGraphTensor* a,
                                                MPSGraphTensor* b) {
        return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    // beta * self (broadcast) + alpha * (sparse @ dense)
    Tensor scaled_self = binaryOp(self, beta_t, mulBlock);
    Tensor scaled_prod = binaryOp(prod, alpha_t, mulBlock);
    return binaryOp(scaled_self, scaled_prod, addBlock);
}

Tensor coalesce_sparse_mps(const Tensor& self) {
    return coalesce_kernel(self);
}

TENSORPLAY_LIBRARY_IMPL(MPS, SparseKernelsMPS) {
    m.impl("sparse_coo_tensor", sparse_coo_tensor_kernel);
    m.impl("to_sparse", to_sparse_kernel);
    m.impl("to_sparse.sparse_dim", to_sparse_sparse_dim_kernel);
    m.impl("_sparse_addmm", sparse_addmm_kernel);
}

TENSORPLAY_LIBRARY_IMPL(Sparse, SparseKernelsSparse) {
    m.impl("to_dense", to_dense_kernel);
    m.impl("_nnz", nnz_kernel);
    m.impl("_coalesce", coalesce_kernel);
    m.impl("coalesce", coalesce_kernel);
    m.impl("sparse_mm", sparse_mm_kernel);
    m.impl("sparse_add", sparse_add_kernel);
    m.impl("sparse_mul", sparse_mul_kernel);
    m.impl("sparse_sum", sparse_sum_kernel);
    m.impl("_sparse_sum", _sparse_sum_kernel);
    m.impl("_sparse_sum.dtype", _sparse_sum_dtype_kernel);
    m.impl("_sparse_sum.dim", _sparse_sum_dim_kernel);
    m.impl("_sparse_sum.dim_dtype", _sparse_sum_dim_dtype_kernel);
}

} // namespace mps
} // namespace tensorplay