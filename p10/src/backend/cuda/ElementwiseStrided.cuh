#pragma once

#include "Tensor.h"
#include "TensorIterator.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "IntegerDivider.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <utility>
#include <vector>

namespace tensorplay {
namespace cuda {

// Strided elementwise machinery for kernels that must read or update an
// operand through an arbitrary dense layout (views, permutations, slices)
// without materializing a contiguous copy first.  The dimension order and
// per-dimension extent come from a TensorIterator pass over the operands,
// which reorders dimensions fastest-first and coalesces adjacent compatible
// dimensions, so the per-element offset reduces to a short div/mod chain.
// Layouts whose coalesced rank exceeds kStridedMaxDims cannot use the fast
// path; callers fall back to a contiguous copy in that case.
constexpr int kStridedMaxDims = 8;

struct StridedDims {
    int ndim = 0;
    int64_t size[kStridedMaxDims] = {};
    int64_t out_stride[kStridedMaxDims] = {};  // in elements
    int64_t in_stride[kStridedMaxDims] = {};   // in elements
};

// Same iteration space with the per-dimension extents converted to invariant
// dividers; the device walk then replaces the generic div/mod pair with one
// multiply-shift per dimension.  Strides stay 64-bit so the accumulated
// element offsets remain valid for tensors beyond the 32-bit range.
struct StridedDimsDiv {
    int ndim = 0;
    detail::IntDividerU32 size[kStridedMaxDims];
    int64_t out_stride[kStridedMaxDims] = {};
    int64_t in_stride[kStridedMaxDims] = {};
};

// Both views describe the same layout; the divider form is eligible only
// when every extent and the element count fit the fast divider's range.
inline bool make_strided_dims_div(const StridedDims& dims,
                                  int64_t n, StridedDimsDiv* out) {
    if (n > static_cast<int64_t>(0x7fffffffLL)) return false;
    out->ndim = dims.ndim;
    for (int d = 0; d < dims.ndim; ++d) {
        const int64_t sz = dims.size[d];
        if (sz < 1 || sz > 0x7fffffffLL) return false;
        out->size[d] = detail::IntDividerU32(static_cast<uint32_t>(sz));
        out->out_stride[d] = dims.out_stride[d];
        out->in_stride[d] = dims.in_stride[d];
    }
    return true;
}

// Extracts the coalesced iteration space of a two-operand iterator
// (operand 0 = output, operand 1 = input).  Strides arrive in bytes and are
// normalized to element units for typed pointers.  Returns false when the
// coalesced rank exceeds kStridedMaxDims.
inline bool fill_strided_dims(
    const TensorIterator& iter, int64_t elem_out, int64_t elem_in,
    StridedDims* dims) {
    const int ndim = iter.ndim();
    if (ndim > kStridedMaxDims) return false;
    dims->ndim = ndim;
    for (int d = 0; d < ndim; ++d) {
        dims->size[d] = iter.shape()[d];
        dims->out_stride[d] = iter.strides(0)[d] / elem_out;
        dims->in_stride[d] = iter.strides(1)[d] / elem_in;
    }
    return true;
}

inline void elementwise_grid(int64_t n, dim3* grid, dim3* block) {
    block->x = 256;
    const int64_t want = (n + 255) / 256;
    grid->x = static_cast<unsigned>(want < 1 ? 1 : want);
}

// out[o(i)] = func(in[v(i)]); o/v are the linear element offsets of output
// and input for the flattened output coordinate i.  A size-1 dimension
// contributes no offset (broadcast or kept singleton dimension).
template <typename T, typename Func>
__global__ void elementwise_strided_kernel_cuda_impl(
    int64_t n, T* out, const T* in, StridedDims dims, Func func) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t gs = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += gs) {
        int64_t o = 0, v = 0, rem = i;
        for (int d = 0; d < dims.ndim; ++d) {
            const int64_t c = rem % dims.size[d];
            rem /= dims.size[d];
            o += c * dims.out_stride[d];
            v += c * dims.in_stride[d];
        }
        out[o] = func(in[v]);
    }
}

// Divider-based twin of the kernel above: the coordinate walk runs on
// 32-bit divmods (one multiply-shift each) and only the offset accumulation
// stays 64-bit.
template <typename T, typename Func>
__global__ void elementwise_strided_kernel_div_cuda_impl(
    int64_t n, T* out, const T* in, StridedDimsDiv dims, Func func) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t gs = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += gs) {
        int64_t o = 0, v = 0;
        uint32_t rem = static_cast<uint32_t>(i);
        for (int d = 0; d < dims.ndim; ++d) {
            const detail::DivModU32 dm = dims.size[d].divmod(rem);
            rem = dm.div;
            o += static_cast<int64_t>(dm.mod) * dims.out_stride[d];
            v += static_cast<int64_t>(dm.mod) * dims.in_stride[d];
        }
        out[o] = func(in[v]);
    }
}

// x[v(i)] = func(x[v(i)]) with one read-modify-write per element; the output
// and input layouts coincide, so no cross-element hazard can arise.
template <typename T, typename Func>
__global__ void elementwise_strided_inplace_kernel_cuda_impl(
    int64_t n, T* x, StridedDims dims, Func func) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t gs = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += gs) {
        int64_t v = 0, rem = i;
        for (int d = 0; d < dims.ndim; ++d) {
            const int64_t c = rem % dims.size[d];
            rem /= dims.size[d];
            v += c * dims.in_stride[d];
        }
        x[v] = func(x[v]);
    }
}

template <typename T, typename Func>
__global__ void elementwise_strided_inplace_kernel_div_cuda_impl(
    int64_t n, T* x, StridedDimsDiv dims, Func func) {
    int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const int64_t gs = static_cast<int64_t>(blockDim.x) * gridDim.x;
    for (; i < n; i += gs) {
        int64_t v = 0;
        uint32_t rem = static_cast<uint32_t>(i);
        for (int d = 0; d < dims.ndim; ++d) {
            const detail::DivModU32 dm = dims.size[d].divmod(rem);
            rem = dm.div;
            v += static_cast<int64_t>(dm.mod) * dims.in_stride[d];
        }
        x[v] = func(x[v]);
    }
}

// Launches the divider-based kernel when the iteration space fits the fast
// divider range; returns false so the caller can fall back to the generic
// 64-bit kernel.
template <typename KernelU32, typename Func, typename... Args>
inline bool launch_div_kernel(int64_t n, const StridedDims& dims,
                              KernelU32 kernel, Func func, Args... args) {
    StridedDimsDiv dims_div;
    if (!make_strided_dims_div(dims, n, &dims_div)) return false;
    dim3 grid, block;
    elementwise_grid(n, &grid, &block);
    kernel<<<grid, block, 0, getCurrentCUDAStream().stream()>>>(
        n, args..., dims_div, func);
    return true;
}

namespace detail {

inline void check_launch(const char* what) {
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        TP_THROW(RuntimeError, std::string(what) + ": " + cudaGetErrorString(err));
    }
}

// Applies func once per unique element of self on a contiguous
// materialization, then writes the values back through the (possibly
// overlapping) view.  This is the safe semantics for layouts where a plain
// read-modify-write walk would touch one address more than once.
template <typename T, typename Func>
void apply_inplace_materialized(Tensor& self, Func func) {
    Tensor tmp = self.contiguous();
    const int64_t n = tmp.numel();
    if (n > 0) {
        StridedDims dims;
        dims.ndim = 1;
        dims.size[0] = n;
        dims.out_stride[0] = 1;
        dims.in_stride[0] = 1;
        dim3 grid, block;
        elementwise_grid(n, &grid, &block);
        elementwise_strided_inplace_kernel_cuda_impl<T><<<grid, block, 0,
            getCurrentCUDAStream().stream()>>>(n, tmp.data_ptr<T>(), dims, func);
        check_launch("elementwise_strided_inplace_kernel");
    }
    self.copy_(tmp);
}

// Sufficient non-overlap condition for an in-place RMW walk: with dims of
// size > 1 sorted by ascending stride, every stride must clear the memory
// extent covered by all smaller-stride dims.  Rejects broadcast (zero-stride)
// dims and overlapping as_strided windows, where one write coordinate would
// alias another.
inline bool is_non_overlapping_strided(const Tensor& t) {
    std::vector<std::pair<int64_t, int64_t>> dims;  // (stride, size)
    const int ndim = t.dim();
    for (int d = 0; d < ndim; ++d) {
        const int64_t sz = t.size(d);
        if (sz > 1) dims.emplace_back(t.stride(d), sz);
    }
    std::sort(dims.begin(), dims.end());
    int64_t reach = 1;
    for (const auto& d : dims) {
        if (d.first < reach) return false;
        reach = d.first * d.second;
    }
    return true;
}

}  // namespace detail

// Applies func elementwise from strided `self` into the freshly allocated
// contiguous `result`.  Returns false only when the coalesced rank exceeds
// kStridedMaxDims; the caller then materializes self.contiguous() and runs
// its contiguous kernel.  A contiguous self is also reported as false so the
// caller keeps its vectorized fast path.
template <typename T, typename Func>
bool launch_unary_strided(const Tensor& self, Tensor& result, Func func) {
    const int64_t n = self.numel();
    if (n == 0 || self.is_contiguous()) return false;
    TensorIterator iter = TensorIteratorConfig()
        .add_owned_output(result)
        .add_owned_const_input(self)
        .build();
    StridedDims dims;
    if (!fill_strided_dims(iter, sizeof(T), sizeof(T), &dims)) return false;
    dim3 grid, block;
    elementwise_grid(n, &grid, &block);
    if (!launch_div_kernel(
            n, dims, elementwise_strided_kernel_div_cuda_impl<T, Func>, func,
            result.data_ptr<T>(), self.data_ptr<T>())) {
        elementwise_strided_kernel_cuda_impl<T><<<grid, block, 0,
            getCurrentCUDAStream().stream()>>>(
            n, result.data_ptr<T>(), self.data_ptr<T>(), dims, func);
    }
    detail::check_launch("elementwise_strided_kernel");
    return true;
}

// Applies func elementwise in place on strided `self`.  Contiguous input is
// reported as false so the caller keeps its vectorized fast path; overlapping
// layouts (broadcast views, as_strided windows) are applied through a
// contiguous materialization so each element is transformed exactly once; a
// false return otherwise means the coalesced rank exceeds kStridedMaxDims and
// the caller must materialize too.
template <typename T, typename Func>
bool launch_unary_inplace_strided(Tensor& self, Func func) {
    const int64_t n = self.numel();
    if (n == 0 || self.is_contiguous()) return false;
    if (!detail::is_non_overlapping_strided(self)) {
        detail::apply_inplace_materialized<T>(self, func);
        return true;
    }
    TensorIterator iter = TensorIteratorConfig()
        .add_owned_output(self)
        .add_owned_const_input(self)
        .build();
    StridedDims dims;
    if (!fill_strided_dims(iter, sizeof(T), sizeof(T), &dims)) {
        detail::apply_inplace_materialized<T>(self, func);
        return true;
    }
    dim3 grid, block;
    elementwise_grid(n, &grid, &block);
    if (!launch_div_kernel(
            n, dims, elementwise_strided_inplace_kernel_div_cuda_impl<T, Func>,
            func, self.data_ptr<T>())) {
        elementwise_strided_inplace_kernel_cuda_impl<T><<<grid, block, 0,
            getCurrentCUDAStream().stream()>>>(n, self.data_ptr<T>(), dims, func);
    }
    detail::check_launch("elementwise_strided_inplace_kernel");
    return true;
}

}  // namespace cuda
}  // namespace tensorplay
