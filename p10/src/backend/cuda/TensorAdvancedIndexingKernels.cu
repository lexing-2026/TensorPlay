// Advanced index selection operators - CUDA kernels.
#include "Tensor.h"
#include "Dispatcher.h"
#include "Scalar.h"
#include "Exception.h"
#include "Utils.h"
#include "CUDARuntime.h"
#include "CUDALoops.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace tensorplay {
namespace cuda {

#define CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      TP_THROW(RuntimeError, std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)

Tensor gather_cuda(const Tensor& self, int64_t dim, const Tensor& index);
Tensor nonzero_cuda(const Tensor& self);

namespace {

constexpr int kAdvancedIndexMaxDims = 16;

struct AdvancedIndexInfo {
    Tensor source;
    std::vector<Tensor> indices;
    std::vector<int64_t> indexed_sizes;
    std::vector<int64_t> indexed_strides;
    std::vector<int64_t> output_shape;
};

inline bool is_index_dtype(DType dtype) {
    return dtype == DType::Int32 || dtype == DType::Int64 ||
           dtype == DType::UInt8 || dtype == DType::Bool;
}

inline bool has_contiguous_index_subspace(const std::vector<Tensor>& indices) {
    bool seen = false;
    bool ended = false;
    for (const Tensor& index : indices) {
        if (!index.defined()) {
            if (seen) ended = true;
        } else {
            if (ended) return false;
            seen = true;
        }
    }
    return true;
}

AdvancedIndexInfo prepare_advanced_index(
        const Tensor& self,
        const std::vector<std::optional<Tensor>>& original_indices) {
    if (original_indices.empty()) {
        TP_THROW(IndexError, "index: at least one index must be provided");
    }
    if (original_indices.size() > static_cast<size_t>(self.dim())) {
        TP_THROW(IndexError,
                 "too many indices for tensor of dimension ", self.dim(),
                 " (got ", original_indices.size(), ")");
    }

    std::vector<Tensor> indices;
    indices.reserve(static_cast<size_t>(self.dim()));
    int64_t input_dim = 0;
    for (const auto& index_opt : original_indices) {
        if (!index_opt.has_value() || !index_opt->defined()) {
            indices.emplace_back();
            ++input_dim;
            continue;
        }

        Tensor index = *index_opt;
        if (!is_index_dtype(index.dtype())) {
            TP_THROW(IndexError,
                     "tensors used as indices must be long, int, byte or bool tensors");
        }
        if (index.device() != self.device()) {
            index = index.to(self.device());
        }

        if (index.dtype() == DType::Bool || index.dtype() == DType::UInt8) {
            const int64_t mask_dim = index.dim();
            if (input_dim + mask_dim > self.dim()) {
                TP_THROW(IndexError,
                         "The shape of the mask does not match the indexed tensor");
            }
            for (int64_t d = 0; d < mask_dim; ++d) {
                if (index.size(d) != self.size(input_dim + d)) {
                    TP_THROW(IndexError,
                             "The shape of the mask does not match the indexed tensor");
                }
            }
            Tensor coordinates = nonzero_cuda(index);
            for (int64_t d = 0; d < mask_dim; ++d) {
                indices.emplace_back(coordinates.select(1, d));
            }
            input_dim += mask_dim;
        } else {
            if (index.dtype() == DType::Int32) {
                index = index.to(DType::Int64);
            }
            indices.emplace_back(index);
            ++input_dim;
        }
    }
    if (input_dim > self.dim()) {
        TP_THROW(IndexError,
                 "too many indices for tensor of dimension ", self.dim());
    }
    while (indices.size() < static_cast<size_t>(self.dim())) {
        indices.emplace_back();
    }

    std::vector<int64_t> advanced_shape;
    bool has_advanced = false;
    for (const Tensor& index : indices) {
        if (!index.defined()) continue;
        if (!has_advanced) {
            advanced_shape = static_cast<std::vector<int64_t>>(index.shape());
            has_advanced = true;
        } else {
            advanced_shape = broadcast_shapes(
                advanced_shape,
                static_cast<std::vector<int64_t>>(index.shape()));
        }
    }
    if (!has_advanced) {
        return AdvancedIndexInfo{self, {}, {}, {},
                                 static_cast<std::vector<int64_t>>(self.shape())};
    }
    for (Tensor& index : indices) {
        if (index.defined() &&
            static_cast<std::vector<int64_t>>(index.shape()) != advanced_shape) {
            index = index.expand(advanced_shape);
        }
    }

    Tensor source = self;
    if (!has_contiguous_index_subspace(indices)) {
        std::vector<int64_t> permutation;
        std::vector<Tensor> reordered;
        permutation.reserve(indices.size());
        reordered.reserve(indices.size());
        for (int64_t d = 0; d < self.dim(); ++d) {
            if (indices[static_cast<size_t>(d)].defined()) {
                permutation.push_back(d);
                reordered.emplace_back(indices[static_cast<size_t>(d)]);
            }
        }
        for (int64_t d = 0; d < self.dim(); ++d) {
            if (!indices[static_cast<size_t>(d)].defined()) {
                permutation.push_back(d);
                reordered.emplace_back(indices[static_cast<size_t>(d)]);
            }
        }
        source = source.permute(permutation);
        indices = std::move(reordered);
    }

    int64_t first_index_dim = self.dim();
    int64_t last_index_dim = -1;
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (indices[static_cast<size_t>(d)].defined()) {
            first_index_dim = std::min(first_index_dim, d);
            last_index_dim = std::max(last_index_dim, d);
        }
    }
    const int64_t dims_before = first_index_dim;
    const int64_t dims_after = self.dim() - last_index_dim - 1;
    const int64_t dims_indexed = last_index_dim - first_index_dim + 1;

    std::vector<int64_t> indexed_sizes;
    std::vector<int64_t> indexed_strides;
    indexed_sizes.reserve(static_cast<size_t>(dims_indexed));
    indexed_strides.reserve(static_cast<size_t>(dims_indexed));
    for (int64_t d = first_index_dim; d <= last_index_dim; ++d) {
        if (!indices[static_cast<size_t>(d)].defined()) continue;
        indexed_sizes.push_back(source.size(d));
        indexed_strides.push_back(
            source.stride(d) * static_cast<int64_t>(elementSize(source.dtype())));
    }

    std::vector<int64_t> source_shape =
        static_cast<std::vector<int64_t>>(source.shape());
    std::vector<int64_t> source_strides =
        static_cast<std::vector<int64_t>>(source.strides());
    source_shape.erase(source_shape.begin() + first_index_dim,
                       source_shape.begin() + first_index_dim + dims_indexed);
    source_strides.erase(source_strides.begin() + first_index_dim,
                         source_strides.begin() + first_index_dim + dims_indexed);
    source_shape.insert(source_shape.begin() + first_index_dim,
                        advanced_shape.begin(), advanced_shape.end());
    source_strides.insert(source_strides.begin() + first_index_dim,
                          advanced_shape.size(), 0);
    source = source.as_strided(source_shape, source_strides);

    std::vector<int64_t> index_shape(static_cast<size_t>(dims_before), 1);
    index_shape.insert(index_shape.end(), advanced_shape.begin(), advanced_shape.end());
    index_shape.insert(index_shape.end(), static_cast<size_t>(dims_after), 1);
    std::vector<Tensor> materialized_indices;
    materialized_indices.reserve(indexed_sizes.size());
    for (Tensor& index : indices) {
        if (index.defined()) {
            materialized_indices.emplace_back(index.reshape(index_shape));
        }
    }

    return AdvancedIndexInfo{
        source,
        std::move(materialized_indices),
        std::move(indexed_sizes),
        std::move(indexed_strides),
        std::move(source_shape)};
}

template <int NumIndices, typename scalar_t>
void launch_advanced_index_kernel(const AdvancedIndexInfo& info, Tensor& output) {
    static_assert(NumIndices > 0);
    TP_CHECK(static_cast<int>(output.dim()) <= kAdvancedIndexMaxDims,
             "index: tensor rank exceeds CUDA indexing limit");

    const int64_t rank = output.dim();
    const auto output_shape = static_cast<std::vector<int64_t>>(output.shape());
    std::array<std::vector<int64_t>, NumIndices + 1> byte_strides;
    byte_strides[0].resize(static_cast<size_t>(rank));
    for (int64_t d = 0; d < rank; ++d) {
        byte_strides[0][static_cast<size_t>(d)] =
            info.source.stride(d) * static_cast<int64_t>(sizeof(scalar_t));
    }
    for (int i = 0; i < NumIndices; ++i) {
        byte_strides[static_cast<size_t>(i + 1)].resize(static_cast<size_t>(rank));
        for (int64_t d = 0; d < rank; ++d) {
            byte_strides[static_cast<size_t>(i + 1)][static_cast<size_t>(d)] =
                info.indices[static_cast<size_t>(i)].stride(d) *
                static_cast<int64_t>(sizeof(int64_t));
        }
    }

    std::array<const int64_t*, NumIndices + 1> stride_ptrs{};
    bool fast_offsets = output.numel() <=
        static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
    for (int i = 0; i < NumIndices + 1; ++i) {
        stride_ptrs[static_cast<size_t>(i)] =
            byte_strides[static_cast<size_t>(i)].data();
        for (int64_t stride : byte_strides[static_cast<size_t>(i)]) {
            fast_offsets = fast_offsets && stride >= 0 &&
                stride <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max());
        }
    }

    std::array<const int64_t*, NumIndices> index_ptrs{};
    for (int i = 0; i < NumIndices; ++i) {
        index_ptrs[static_cast<size_t>(i)] =
            info.indices[static_cast<size_t>(i)].data_ptr<int64_t>();
    }
    const scalar_t* source_ptr = info.source.data_ptr<scalar_t>();
    std::array<int64_t, NumIndices> indexed_sizes{};
    std::array<int64_t, NumIndices> indexed_strides{};
    for (int i = 0; i < NumIndices; ++i) {
        indexed_sizes[static_cast<size_t>(i)] = info.indexed_sizes[static_cast<size_t>(i)];
        indexed_strides[static_cast<size_t>(i)] = info.indexed_strides[static_cast<size_t>(i)];
    }

    if (fast_offsets) {
        OffsetCalculator<NumIndices + 1, uint32_t> offsets(
            static_cast<int>(rank), output_shape.data(), stride_ptrs.data());
        gpu_kernel_with_index(output, [=] GPU_LAMBDA(int64_t linear_index) -> scalar_t {
            const auto byte_offsets = offsets.get(static_cast<uint32_t>(linear_index));
            int64_t source_offset = static_cast<int64_t>(byte_offsets[0]);
#pragma unroll
            for (int i = 0; i < NumIndices; ++i) {
                int64_t index = *reinterpret_cast<const int64_t*>(
                    index_ptrs[static_cast<size_t>(i)] + byte_offsets[static_cast<size_t>(i + 1)]);
                assert(index >= -indexed_sizes[static_cast<size_t>(i)] &&
                       index < indexed_sizes[static_cast<size_t>(i)]);
                if (index < 0) index += indexed_sizes[static_cast<size_t>(i)];
                source_offset += index * indexed_strides[static_cast<size_t>(i)];
            }
            return *reinterpret_cast<const scalar_t*>(
                reinterpret_cast<const char*>(source_ptr) + source_offset);
        });
        return;
    }

    std::array<int64_t, kAdvancedIndexMaxDims> shape{};
    std::array<int64_t, kAdvancedIndexMaxDims> source_byte_strides{};
    std::array<std::array<int64_t, kAdvancedIndexMaxDims>, NumIndices>
        index_byte_strides{};
    for (int64_t d = 0; d < rank; ++d) {
        shape[static_cast<size_t>(d)] = output_shape[static_cast<size_t>(d)];
        source_byte_strides[static_cast<size_t>(d)] = byte_strides[0][static_cast<size_t>(d)];
        for (int i = 0; i < NumIndices; ++i) {
            index_byte_strides[static_cast<size_t>(i)][static_cast<size_t>(d)] =
                byte_strides[static_cast<size_t>(i + 1)][static_cast<size_t>(d)];
        }
    }
    gpu_kernel_with_index(output, [=] GPU_LAMBDA(int64_t linear_index) -> scalar_t {
        int64_t remainder = linear_index;
        int64_t source_offset = 0;
        int64_t index_offsets[NumIndices] = {};
        for (int64_t d = rank - 1; d >= 0; --d) {
            const int64_t coordinate = remainder % shape[static_cast<size_t>(d)];
            remainder /= shape[static_cast<size_t>(d)];
            source_offset += coordinate * source_byte_strides[static_cast<size_t>(d)];
            for (int i = 0; i < NumIndices; ++i) {
                index_offsets[i] += coordinate *
                    index_byte_strides[static_cast<size_t>(i)][static_cast<size_t>(d)];
            }
        }
        for (int i = 0; i < NumIndices; ++i) {
            int64_t index = *reinterpret_cast<const int64_t*>(
                index_ptrs[static_cast<size_t>(i)] + index_offsets[i]);
            assert(index >= -indexed_sizes[static_cast<size_t>(i)] &&
                   index < indexed_sizes[static_cast<size_t>(i)]);
            if (index < 0) index += indexed_sizes[static_cast<size_t>(i)];
            source_offset += index * indexed_strides[static_cast<size_t>(i)];
        }
        return *reinterpret_cast<const scalar_t*>(
            reinterpret_cast<const char*>(source_ptr) + source_offset);
    });
}

template <int NumIndices>
void dispatch_advanced_index_kernel(const AdvancedIndexInfo& info, Tensor& output) {
#define TP_ADVANCED_INDEX_CASE(ctype, name) \
    case DType::name: \
        launch_advanced_index_kernel<NumIndices, ctype>(info, output); \
        break;
    switch (info.source.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_ADVANCED_INDEX_CASE)
        TENSORPLAY_FORALL_FP8_TYPES(TP_ADVANCED_INDEX_CASE)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<Half>, ComplexHalf)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<float>, ComplexFloat)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<double>, ComplexDouble)
        TP_ADVANCED_INDEX_CASE(tensorplay::complex<BFloat16>, BComplex32)
        default:
            TP_THROW(TypeError, "index: unsupported dtype");
    }
#undef TP_ADVANCED_INDEX_CASE
}

Tensor index_cuda(const Tensor& self,
                  const std::vector<std::optional<Tensor>>& indices) {
    AdvancedIndexInfo info = prepare_advanced_index(self, indices);
    if (info.indices.empty()) return self;
    Tensor output = Tensor::empty(info.output_shape, self.dtype(), self.device());
    if (output.numel() == 0) return output;
    switch (info.indices.size()) {
        case 1: dispatch_advanced_index_kernel<1>(info, output); break;
        case 2: dispatch_advanced_index_kernel<2>(info, output); break;
        case 3: dispatch_advanced_index_kernel<3>(info, output); break;
        case 4: dispatch_advanced_index_kernel<4>(info, output); break;
        case 5: dispatch_advanced_index_kernel<5>(info, output); break;
        case 6: dispatch_advanced_index_kernel<6>(info, output); break;
        case 7: dispatch_advanced_index_kernel<7>(info, output); break;
        case 8: dispatch_advanced_index_kernel<8>(info, output); break;
        case 9: dispatch_advanced_index_kernel<9>(info, output); break;
        case 10: dispatch_advanced_index_kernel<10>(info, output); break;
        case 11: dispatch_advanced_index_kernel<11>(info, output); break;
        case 12: dispatch_advanced_index_kernel<12>(info, output); break;
        case 13: dispatch_advanced_index_kernel<13>(info, output); break;
        case 14: dispatch_advanced_index_kernel<14>(info, output); break;
        case 15: dispatch_advanced_index_kernel<15>(info, output); break;
        case 16: dispatch_advanced_index_kernel<16>(info, output); break;
        default:
            TP_THROW(IndexError, "index: too many advanced index tensors");
    }
    CUDA_CHECK(cudaGetLastError());
    return output;
}

inline int64_t wrap_dim(int64_t dim, int64_t ndim) {
    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        TP_THROW(RuntimeError, "Dimension out of range (expected to be in range of [",
                 -ndim, ", ", ndim - 1, "], but got ", dim - ndim, ")");
    }
    return dim;
}

Tensor take_along_dim_cuda(const Tensor& self, const Tensor& indices, std::optional<int64_t> dim) {
    if (indices.dtype() != DType::Int64) {
        TP_THROW(TypeError, "take_along_dim: expected indices to have dtype Int64");
    }
    if (self.device() != indices.device()) {
        TP_THROW(DeviceMismatchError,
                 "take_along_dim: self and indices must be on the same device");
    }
    if (!dim.has_value()) {
        Tensor flat = self.view({-1});
        Tensor idx = indices.view({-1});
        return gather_cuda(flat, 0, idx);
    }
    int64_t nd = self.dim();
    int64_t d = wrap_dim(*dim, nd);
    if (indices.dim() != nd) {
        TP_THROW(RuntimeError, "take_along_dim: indices must have the same number of dimensions as input");
    }
    std::vector<int64_t> target(nd);
    for (int64_t i = 0; i < nd; ++i) {
        if (i == d) { target[i] = indices.size(i); continue; }
        int64_t a = self.size(i), b = indices.size(i);
        if (a != b && a != 1 && b != 1) {
            TP_THROW(RuntimeError, "take_along_dim: input and indices must match on non-selected dimensions");
        }
        target[i] = std::max(a, b);
    }
    std::vector<int64_t> idx_target = target;
    std::vector<int64_t> self_target = target;
    self_target[d] = self.size(d);
    Tensor idx_b = indices.expand(idx_target).contiguous();
    Tensor self_b = self.expand(self_target).contiguous();
    idx_b = idx_b.remainder(Scalar(self_b.size(d)));
    return gather_cuda(self_b, d, idx_b);
}


} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, TensorAdvancedIndexingKernels) {
    m.impl("index.Tensor", index_cuda);
    m.impl("take_along_dim", take_along_dim_cuda);
}

} // namespace cuda
} // namespace tensorplay
