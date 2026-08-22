#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "CUDARuntime.h"
#include "Allocator.h"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include <algorithm>

namespace tensorplay {
namespace cuda {

// Non-constant padding modes (torch: reflection_padNd / replication_padNd /
// circular_padNd).  Index math mirrors p10/src/backend/cpu/PadKernels.cpp,
// which is verified bit-exact against torch's F.pad for all three modes.

namespace {

enum class PadIndexMode { Reflect = 0, Replicate = 1, Circular = 2 };

struct PadPlan {
    int ndim;
    int k_padded;
    int pad_pairs[8];      // torch order: [l(last), r(last), ...]
    int src_sizes[8];
    long long src_strides[8];
    int dst_sizes[8];
    long long dst_strides[8];
};

__device__ inline long long pad_map_coord(long long q, long long left, long long size,
                                          int mode) {
    long long src = q - left;
    if (src >= 0 && src < size) return src;
    if (mode == 0) {          // reflect: single mirror (pad < size enforced host-side)
        if (src < 0) src = -src;
        else src = 2 * size - 2 - src;
        return src;
    }
    if (mode == 1) {          // replicate: clamp
        return min(max(src, 0LL), size - 1);
    }
    // circular: python-style positive modulo (pad <= size enforced host-side,
    // matching torch's "wrapping around more than once" check)
    return ((src % size) + size) % size;
}

template <typename T>
__global__ void pad_gather_kernel(const T* __restrict__ src, T* __restrict__ dst,
                                  PadPlan p, int mode) {
    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long rem = idx;
    long long src_off = 0;
    for (int d = 0; d < p.ndim; ++d) {
        const long long q = rem / p.dst_strides[d];
        rem -= q * p.dst_strides[d];
        long long s = q;
        if (d >= p.ndim - p.k_padded) {
            const int pair = p.ndim - 1 - d;
            s = pad_map_coord(q, p.pad_pairs[2 * pair], p.src_sizes[d], mode);
        }
        src_off += s * p.src_strides[d];
    }
    dst[idx] = src[src_off];
}

// Backward: scatter-add each padded position into grad_input.  float and
// double atomics are both native on sm_60+ (the build targets sm_86).
template <typename T>
__global__ void pad_scatter_kernel(const T* __restrict__ go, T* __restrict__ gi,
                                   PadPlan p, int mode) {
    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    long long rem = idx;
    long long src_off = 0;
    for (int d = 0; d < p.ndim; ++d) {
        const long long q = rem / p.dst_strides[d];
        rem -= q * p.dst_strides[d];
        long long s = q;
        if (d >= p.ndim - p.k_padded) {
            const int pair = p.ndim - 1 - d;
            s = pad_map_coord(q, p.pad_pairs[2 * pair], p.src_sizes[d], mode);
        }
        src_off += s * p.src_strides[d];
    }
    atomicAdd(gi + src_off, go[idx]);
}

PadPlan make_plan(const Tensor& self, const std::vector<int64_t>& pad) {
    PadPlan p{};
    p.ndim = static_cast<int>(self.dim());
    p.k_padded = static_cast<int>(pad.size()) / 2;
    std::vector<int64_t> src_strides(p.ndim, 1);
    for (int64_t d = p.ndim - 2; d >= 0; --d)
        src_strides[d] = src_strides[d + 1] * self.size(d + 1);
    std::vector<int64_t> dst_sizes = self.shape();
    for (int64_t i = 0; i < p.k_padded; ++i)
        dst_sizes[p.ndim - 1 - i] += pad[2 * i] + pad[2 * i + 1];
    std::vector<int64_t> dst_strides(p.ndim, 1);
    for (int64_t d = p.ndim - 2; d >= 0; --d)
        dst_strides[d] = dst_strides[d + 1] * dst_sizes[d + 1];
    for (int i = 0; i < p.k_padded; ++i) {
        p.pad_pairs[2 * i] = static_cast<int>(pad[2 * i]);
        p.pad_pairs[2 * i + 1] = static_cast<int>(pad[2 * i + 1]);
    }
    for (int d = 0; d < p.ndim; ++d) {
        p.src_sizes[d] = static_cast<int>(self.size(d));
        p.src_strides[d] = src_strides[d];
        p.dst_sizes[d] = static_cast<int>(dst_sizes[d]);
        p.dst_strides[d] = dst_strides[d];
    }
    return p;
}

void pad_mode_check(const char* name, const Tensor& self, const std::vector<int64_t>& pad,
                    int mode) {
    if (pad.size() % 2 != 0)
        TP_THROW(ValueError, name, ": length of pad must be even but it equals ", pad.size());
    const int64_t k = static_cast<int64_t>(pad.size()) / 2;
    if (k > 4 || k > self.dim())
        TP_THROW(ValueError, name, ": padding length too large for input dimensionality");
    for (int64_t i = 0; i < k; ++i) {
        const int64_t dim = self.dim() - 1 - i;
        if (mode == 0) {
            if (pad[2 * i] >= self.size(dim) || pad[2 * i + 1] >= self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding size should be less than the corresponding input dimension");
        } else if (mode == 2) {
            if (pad[2 * i] > self.size(dim) || pad[2 * i + 1] > self.size(dim))
                TP_THROW(ValueError, name,
                         ": padding value causes wrapping around more than once");
        }
    }
}

// half/bf16 pads move values without arithmetic, so computing them in float32
// and casting back (torch's CUDA opmath convention) is exact for the forward
// and matches torch's accumulation for the backward.
Tensor pad_mode_forward(const Tensor& self, const std::vector<int64_t>& pad, int mode) {
    Tensor src = self.is_contiguous() ? self : self.contiguous();
    const bool lowp = src.dtype() == DType::Float16 || src.dtype() == DType::BFloat16;
    Tensor work = lowp ? src.to(DType::Float32) : src;
    PadPlan p = make_plan(work, pad);

    std::vector<int64_t> dst_sizes(p.dst_sizes, p.dst_sizes + p.ndim);
    Tensor out = Tensor::empty(dst_sizes, work.dtype(), work.device());

    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (work.dtype() == DType::Float64) {
        pad_gather_kernel<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<double>(), out.data_ptr<double>(), p, mode);
    } else {
        pad_gather_kernel<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            work.data_ptr<float>(), out.data_ptr<float>(), p, mode);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("pad CUDA: ") + cudaGetErrorString(err));
    return lowp ? out.to(src.dtype()) : out;
}

Tensor pad_mode_backward(const Tensor& grad_output, const Tensor& self,
                         const std::vector<int64_t>& pad, int mode) {
    PadPlan p = make_plan(self, pad);
    Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    const bool lowp = go.dtype() == DType::Float16 || go.dtype() == DType::BFloat16;
    Tensor go_work = lowp ? go.to(DType::Float32) : go;
    Tensor gi = Tensor::zeros(self.shape(), go_work.dtype(), self.device());

    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (go_work.dtype() == DType::Float64) {
        pad_scatter_kernel<double><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            go_work.data_ptr<double>(), gi.data_ptr<double>(), p, mode);
    } else {
        pad_scatter_kernel<float><<<blocks, threads, 0, getCurrentCUDAStream().stream()>>>(
            go_work.data_ptr<float>(), gi.data_ptr<float>(), p, mode);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("pad backward CUDA: ") + cudaGetErrorString(err));
    return lowp ? gi.to(go.dtype()) : gi;
}

}  // namespace

Tensor reflection_pad_nd_cuda(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("reflection_pad_nd", self, pad, 0);
    return pad_mode_forward(self, pad, 0);
}

Tensor reflection_pad_nd_backward_cuda(const Tensor& grad_output, const Tensor& self,
                                       const std::vector<int64_t>& pad) {
    pad_mode_check("reflection_pad_nd_backward", self, pad, 0);
    return pad_mode_backward(grad_output, self, pad, 0);
}

Tensor replication_pad_nd_cuda(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("replication_pad_nd", self, pad, 1);
    return pad_mode_forward(self, pad, 1);
}

Tensor replication_pad_nd_backward_cuda(const Tensor& grad_output, const Tensor& self,
                                        const std::vector<int64_t>& pad) {
    pad_mode_check("replication_pad_nd_backward", self, pad, 1);
    return pad_mode_backward(grad_output, self, pad, 1);
}

Tensor circular_pad_nd_cuda(const Tensor& self, const std::vector<int64_t>& pad) {
    pad_mode_check("circular_pad_nd", self, pad, 2);
    return pad_mode_forward(self, pad, 2);
}

Tensor circular_pad_nd_backward_cuda(const Tensor& grad_output, const Tensor& self,
                                     const std::vector<int64_t>& pad) {
    pad_mode_check("circular_pad_nd_backward", self, pad, 2);
    return pad_mode_backward(grad_output, self, pad, 2);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, PadKernels) {
    m.impl("reflection_pad_nd", reflection_pad_nd_cuda);
    m.impl("reflection_pad_nd_backward", reflection_pad_nd_backward_cuda);
    m.impl("replication_pad_nd", replication_pad_nd_cuda);
    m.impl("replication_pad_nd_backward", replication_pad_nd_backward_cuda);
    m.impl("circular_pad_nd", circular_pad_nd_cuda);
    m.impl("circular_pad_nd_backward", circular_pad_nd_backward_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
