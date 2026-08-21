#pragma once

#include "Tensor.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tensorplay {
namespace cuda {

// Broadcast metadata used by the existing CUDA arithmetic and comparison
// kernels.  The output shape is right-aligned with every input shape, just as
// TensorIterator aligns broadcast dimensions.  A singleton input dimension
// gets a zero stride so all output coordinates read the same element.
constexpr int CUDA_BROADCAST_MAX_DIMS = 8;

struct TensorDesc {
    int ndim = 0;
    int64_t sizes[CUDA_BROADCAST_MAX_DIMS]{};
    int64_t strides[CUDA_BROADCAST_MAX_DIMS]{};
};

inline TensorDesc make_desc(const Tensor& tensor, size_t output_ndim) {
    TensorDesc desc;
    if (output_ndim > CUDA_BROADCAST_MAX_DIMS ||
        static_cast<size_t>(tensor.dim()) > output_ndim) {
        throw std::runtime_error("CUDA broadcast dimension exceeds supported rank");
    }

    desc.ndim = static_cast<int>(output_ndim);
    const size_t input_ndim = static_cast<size_t>(tensor.dim());
    const size_t leading = output_ndim - input_ndim;
    for (size_t dim = 0; dim < output_ndim; ++dim) {
        if (dim < leading) {
            desc.sizes[dim] = 1;
            desc.strides[dim] = 0;
            continue;
        }

        const size_t input_dim = dim - leading;
        desc.sizes[dim] = tensor.size(static_cast<int64_t>(input_dim));
        desc.strides[dim] = tensor.stride(static_cast<int64_t>(input_dim));
        if (desc.sizes[dim] == 1) {
            desc.strides[dim] = 0;
        }
    }
    return desc;
}

inline TensorDesc make_desc_from_shape(const std::vector<int64_t>& shape) {
    TensorDesc desc;
    if (shape.size() > CUDA_BROADCAST_MAX_DIMS) {
        throw std::runtime_error("CUDA broadcast dimension exceeds supported rank");
    }

    desc.ndim = static_cast<int>(shape.size());
    int64_t stride = 1;
    for (size_t reverse = shape.size(); reverse > 0; --reverse) {
        const size_t dim = reverse - 1;
        desc.sizes[dim] = shape[dim];
        desc.strides[dim] = stride;
        stride *= std::max<int64_t>(shape[dim], 1);
    }
    return desc;
}

__host__ __device__ inline int64_t logical_stride(
    const TensorDesc& output,
    int dim) {
    int64_t stride = 1;
    for (int next = output.ndim - 1; next > dim; --next) {
        stride *= output.sizes[next] > 0 ? output.sizes[next] : 1;
    }
    return stride;
}

// The CUDA pointwise launch index is in logical contiguous order even when
// the destination uses channels-last physical strides.  This mirrors the
// indexing used by TorchInductor's generated Triton pointwise kernels.
__host__ __device__ inline int64_t get_logical_coordinate(
    int64_t linear_index,
    const TensorDesc& output,
    int dim) {
    const int64_t size = output.sizes[dim];
    if (size <= 1) return 0;
    return (linear_index / logical_stride(output, dim)) % size;
}

__host__ __device__ inline int64_t get_offset(
    int64_t linear_index,
    const TensorDesc& source,
    const TensorDesc& output) {
    int64_t offset = 0;
    for (int dim = output.ndim - 1; dim >= 0; --dim) {
        const int64_t coordinate = get_logical_coordinate(linear_index, output, dim);
        if (source.sizes[dim] != 1) {
            offset += coordinate * source.strides[dim];
        }
    }
    return offset;
}

__host__ __device__ inline int64_t get_output_offset(
    int64_t linear_index,
    const TensorDesc& output) {
    int64_t offset = 0;
    for (int dim = output.ndim - 1; dim >= 0; --dim) {
        offset += get_logical_coordinate(linear_index, output, dim) *
            output.strides[dim];
    }
    return offset;
}

} // namespace cuda
} // namespace tensorplay
