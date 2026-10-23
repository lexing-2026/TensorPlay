#include "QuantTensorOperators.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "Exception.h"
#include "Scalar.h"
#include "CUDARuntime.h"
#include "Allocator.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "Atomic.cuh"

namespace tensorplay {
namespace cuda {

// circular_padNd). Index math supports reflect, replicate, and circular modes,

namespace {

enum class PadIndexMode { Reflect = 0, Replicate = 1, Circular = 2 };

struct PadPlan {
    int ndim;
    int k_padded;
    int pad_pairs[8];
    int src_sizes[8];
    long long src_strides[8];
    int dst_sizes[8];
    long long dst_strides[8];
};

__device__ inline long long pad_map_coord(long long q, long long left, long long size,
                                          int mode) {
    long long src = q - left;
    if (src >= 0 && src < size) return src;
    if (mode == 0) {          // reflect: single reflected index (pad < size enforced host-side)
        if (src < 0) src = -src;
        else src = 2 * size - 2 - src;
        return src;
    }
    if (mode == 1) {          // replicate: clamp
        return min(max(src, 0LL), size - 1);
    }
    // circular: python-style positive modulo (pad <= size enforced host-side,
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

// Constant pad with a zero fill, written in one pass: every output element is
// either the fill or a source element, so the buffer is touched twice (read the
// interior, write everything) instead of the three traversals a fill-then-copy
// pays (write all, read the interior, overwrite the interior).  The interior
// starts `left` elements into each output row, so with an odd pad neither side
// of that copy is 16-byte aligned and the vectorised form of the copy is
// unavailable anyway - dropping the extra traversal is the whole gain.  Zero is
// the all-bits-zero pattern at every storage width, which is what lets this stay
// width-dispatched like the gather kernels above.
//
// RowForm is set when both sides are contiguous and only the innermost dimension
// is padded: every output row is then the source row of the same number shifted
// by `left`, and rows past the end of the source are pure fill.  That costs one
// quotient instead of one per dimension, and the 32-bit form of it stays off the
// emulated 64-bit divide path, whose few hundred instructions per element push
// the walk well under the bus ceiling.
template <typename T, bool RowForm>
__global__ void pad_constant_zero_kernel(const T* __restrict__ src,
                                         T* __restrict__ dst, const PadPlan p,
                                         const unsigned dst_last,
                                         const unsigned src_last,
                                         const unsigned src_rows,
                                         const unsigned left) {
    const T fill = {};
    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    if (RowForm) {
        const unsigned i = static_cast<unsigned>(idx);
        const unsigned row = i / dst_last;
        // Unsigned wrap does the lower-bound test for free: a column left of the
        // interior becomes huge and fails the width comparison.
        const unsigned col = i - row * dst_last - left;
        dst[i] = (row < src_rows && col < src_last)
            ? src[row * src_last + col] : fill;
        return;
    }
    long long rem = idx;
    long long src_off = 0;
    bool interior = true;
    for (int d = 0; d < p.ndim; ++d) {
        const long long q = rem / p.dst_strides[d];
        rem -= q * p.dst_strides[d];
        if (d >= p.ndim - p.k_padded) {
            const int pair = p.ndim - 1 - d;
            const long long s = q - p.pad_pairs[2 * pair];
            if (s < 0 || s >= p.src_sizes[d]) {
                interior = false;
                break;
            }
            src_off += s * p.src_strides[d];
        } else {
            src_off += q * p.src_strides[d];
        }
    }
    dst[idx] = interior ? src[src_off] : fill;
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

Tensor pad_mode_forward(const Tensor& self, const std::vector<int64_t>& pad, int mode) {
    CUDAGuard guard(self.device());
    Tensor src = self.is_contiguous() ? self : self.contiguous();
    PadPlan p = make_plan(src, pad);

    std::vector<int64_t> dst_sizes(p.dst_sizes, p.dst_sizes + p.ndim);
    Tensor out = Tensor::empty(dst_sizes, src.dtype(), src.device());

    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    const int threads = 256;
    const int blocks = static_cast<int>((total + threads - 1) / threads);
    if (total == 0) return out;
    // Padding moves element bit patterns without arithmetic. Dispatch by
    // storage width to preserve integer codes and floating payloads exactly.
    const auto stream = getCurrentCUDAStream().stream();
    switch (elementSize(src.dtype())) {
        case 1: pad_gather_kernel<uint8_t><<<blocks,threads,0,stream>>>(static_cast<const uint8_t*>(src.data_ptr()),static_cast<uint8_t*>(out.data_ptr()),p,mode); break;
        case 2: pad_gather_kernel<uint16_t><<<blocks,threads,0,stream>>>(static_cast<const uint16_t*>(src.data_ptr()),static_cast<uint16_t*>(out.data_ptr()),p,mode); break;
        case 4: pad_gather_kernel<uint32_t><<<blocks,threads,0,stream>>>(static_cast<const uint32_t*>(src.data_ptr()),static_cast<uint32_t*>(out.data_ptr()),p,mode); break;
        case 8: pad_gather_kernel<uint64_t><<<blocks,threads,0,stream>>>(static_cast<const uint64_t*>(src.data_ptr()),static_cast<uint64_t*>(out.data_ptr()),p,mode); break;
        case 16: pad_gather_kernel<ulonglong2><<<blocks,threads,0,stream>>>(static_cast<const ulonglong2*>(src.data_ptr()),static_cast<ulonglong2*>(out.data_ptr()),p,mode); break;
        default: TP_THROW(TypeError, "Unsupported padding storage width");
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("pad CUDA: ") + cudaGetErrorString(err));
    return out;
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

// Launch the one-pass zero fill over the whole output.  Returns false when the
// width has no instantiation or the plan disagrees with the caller's shape, so
// the caller keeps the general fill-and-copy body.
bool constant_pad_zero_forward(const Tensor& self,
                               const std::vector<int64_t>& pad, Tensor& output) {
    if (output.numel() == 0) return true;
    // The walk addresses both sides by flat element index, so it needs the
    // contiguous layout the caller's `empty` is expected to have produced; a
    // permuted output (or a quantized one, whose padded byte is the zero point
    // rather than the all-zero pattern) keeps the general body.
    if (!output.is_contiguous() || isQuantizedType(self.dtype())) return false;
    Tensor src = self.is_contiguous() ? self : self.contiguous();
    const PadPlan p = make_plan(src, pad);
    const long long total = p.ndim > 0
        ? static_cast<long long>(p.dst_strides[0]) * p.dst_sizes[0]
        : 1;
    if (static_cast<long long>(output.numel()) != total) return false;
    // The row form needs a 32-bit index space and an untouched outer shape: a
    // pad on any dimension but the innermost one shifts whole rows, which the
    // single quotient cannot express.
    bool innermost_only = p.ndim >= 1;
    for (int i = 1; i < p.k_padded && innermost_only; ++i) {
        innermost_only = p.pad_pairs[2 * i] == 0 && p.pad_pairs[2 * i + 1] == 0;
    }
    const bool row_form = innermost_only && total <= INT32_MAX;
    const unsigned src_last =
        row_form ? static_cast<unsigned>(p.src_sizes[p.ndim - 1]) : 0u;
    const unsigned dst_last =
        row_form ? static_cast<unsigned>(p.dst_sizes[p.ndim - 1]) : 0u;
    const unsigned src_rows =
        row_form ? static_cast<unsigned>(src.numel() / std::max<int64_t>(src_last, 1)) : 0u;
    const unsigned left =
        row_form ? static_cast<unsigned>(p.pad_pairs[0]) : 0u;
    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((total + threads - 1) / threads);
    const auto stream = getCurrentCUDAStream().stream();
    const void* src_ptr = src.data_ptr();
    void* dst_ptr = output.data_ptr();
    auto launch = [&](auto tag) {
        using T = decltype(tag);
        if (row_form) {
            pad_constant_zero_kernel<T, true><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(src_ptr), static_cast<T*>(dst_ptr), p,
                dst_last, src_last, src_rows, left);
        } else {
            pad_constant_zero_kernel<T, false><<<blocks, threads, 0, stream>>>(
                static_cast<const T*>(src_ptr), static_cast<T*>(dst_ptr), p,
                dst_last, src_last, src_rows, left);
        }
    };
    switch (elementSize(src.dtype())) {
        case 1: launch(uint8_t{}); break;
        case 2: launch(uint16_t{}); break;
        case 4: launch(uint32_t{}); break;
        case 8: launch(uint64_t{}); break;
        case 16: launch(ulonglong2{}); break;
        default: return false;
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        TP_THROW(RuntimeError, std::string("pad CUDA: ") + cudaGetErrorString(err));
    return true;
}

}  // namespace

Tensor reflection_pad_nd_cuda(const Tensor& self, const std::vector<int64_t>& pad) {
    if (isQuantizedType(self.dtype())) { quantized::q_scale(self); return quantized::with_quantizer(reflection_pad_nd_cuda(quantized::storage_view(self), pad), self); }

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

// constant_pad_nd is a composite over slice/fill_/copy_ (see the CPU
// implementation) so the same body serves both backends.
static Tensor constant_pad_nd_cuda(const Tensor& self, const std::vector<int64_t>& pad,
                                   const Scalar& value) {
    auto input_sizes = self.shape();
    int64_t l_inp = self.dim();
    int64_t l_pad = static_cast<int64_t>(pad.size()) / 2;

    if (pad.size() % 2 != 0) {
        TP_THROW(ValueError, "Length of pad must be even but instead it equals ", pad.size());
    }
    if (l_inp < l_pad) {
        TP_THROW(ValueError, "Length of pad should be no more than twice the number of "
                 "dimensions of the input. Pad length is ", pad.size(), " while the input has ",
                 l_inp, " dimensions.");
    }

    bool all_pads_non_positive = true;
    Tensor c_input = self;
    for (int64_t i = l_inp - l_pad; i < l_inp; ++i) {
        int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] < 0) {
            c_input = c_input.slice(i, -pad[pad_idx], c_input.size(i));
        } else if (pad[pad_idx] != 0) {
            all_pads_non_positive = false;
        }
        if (pad[pad_idx + 1] < 0) {
            c_input = c_input.slice(i, 0, c_input.size(i) + pad[pad_idx + 1]);
        } else if (pad[pad_idx + 1] != 0) {
            all_pads_non_positive = false;
        }
    }

    if (all_pads_non_positive) {
        return c_input.clone();
    }

    std::vector<int64_t> new_shape;
    new_shape.reserve(l_inp - l_pad);
    for (int64_t i = 0; i < l_inp - l_pad; ++i) {
        new_shape.push_back(input_sizes[i]);
    }
    for (int64_t i = 0; i < l_pad; ++i) {
        size_t pad_idx = pad.size() - ((i + 1) * 2);
        int64_t new_dim = input_sizes[l_inp - l_pad + i] + pad[pad_idx] + pad[pad_idx + 1];
        if (new_dim < 0) {
            TP_THROW(ValueError, "The input size ", input_sizes[l_inp - l_pad + i],
                     ", plus negative padding ", pad[pad_idx], " and ", pad[pad_idx + 1],
                     " resulted in a negative output size, which is invalid. Check dimension ",
                     l_inp - l_pad + i, " of your input.");
        }
        new_shape.push_back(new_dim);
    }

    Tensor output = Tensor::empty(new_shape, self.dtype(), self.device());

    // One pass over the output whenever the fill is the all-zero bit pattern
    // (a negative zero is not: its sign bit would be lost), every pad is
    // non-negative (a negative one slices the input, so a plan built from self
    // would not describe the region to copy) and the padded rank and input rank
    // fit the plan.  Otherwise fill and copy the interior over the fill.
    const double fill_value =
        (!value.isComplex() && value.dtype() != DType::Undefined)
            ? value.toDouble() : 1.0;
    const bool zero_fill = fill_value == 0.0 && !std::signbit(fill_value);
    const bool plain_pads = l_pad <= 4 && l_inp <= 8 &&
        std::all_of(pad.begin(), pad.end(), [](int64_t p) { return p >= 0; });
    if (zero_fill && plain_pads && constant_pad_zero_forward(self, pad, output)) {
        return output;
    }

    output.fill_(value);

    Tensor c_output = output;
    for (int64_t i = l_inp - l_pad; i < l_inp; ++i) {
        int64_t pad_idx = 2 * (l_inp - i - 1);
        if (pad[pad_idx] > 0) {
            c_output = c_output.slice(i, pad[pad_idx], c_output.size(i));
        }
        if (pad[pad_idx + 1] > 0) {
            c_output = c_output.slice(i, 0, c_output.size(i) - pad[pad_idx + 1]);
        }
    }
    c_output.copy_(c_input);

    return output;
}

static Tensor constant_pad_nd_backward_cuda(const Tensor& grad_output,
                                            const std::vector<int64_t>& pad) {
    std::vector<int64_t> negated_pad = pad;
    for (auto& p : negated_pad) p = -p;
    return constant_pad_nd_cuda(grad_output, negated_pad, Scalar(0));
}

// .grad_input out-variants: run the rank-generic backward and hand the
// result over through the caller's buffer.
#define DEFINE_PAD_BACKWARD_OUT_CUDA(NAME, MODE_KERNEL)                       \
Tensor& NAME##_grad_input_cuda(const Tensor& grad_output, const Tensor& self, \
                               const std::vector<int64_t>& pad,               \
                               Tensor& grad_input) {                          \
    grad_input = MODE_KERNEL(grad_output, self, pad);                         \
    return grad_input;                                                        \
}
DEFINE_PAD_BACKWARD_OUT_CUDA(reflection_pad1d_backward, reflection_pad_nd_backward_cuda)
DEFINE_PAD_BACKWARD_OUT_CUDA(reflection_pad2d_backward, reflection_pad_nd_backward_cuda)
DEFINE_PAD_BACKWARD_OUT_CUDA(reflection_pad3d_backward, reflection_pad_nd_backward_cuda)
DEFINE_PAD_BACKWARD_OUT_CUDA(replication_pad1d_backward, replication_pad_nd_backward_cuda)
DEFINE_PAD_BACKWARD_OUT_CUDA(replication_pad2d_backward, replication_pad_nd_backward_cuda)
DEFINE_PAD_BACKWARD_OUT_CUDA(replication_pad3d_backward, replication_pad_nd_backward_cuda)
#undef DEFINE_PAD_BACKWARD_OUT_CUDA

TENSORPLAY_LIBRARY_IMPL(CUDA, PadKernels) {
    m.impl("reflection_pad_nd", reflection_pad_nd_cuda);
    m.impl("reflection_pad_nd_backward", reflection_pad_nd_backward_cuda);
    m.impl("replication_pad_nd", replication_pad_nd_cuda);
    m.impl("replication_pad_nd_backward", replication_pad_nd_backward_cuda);
    m.impl("circular_pad_nd", circular_pad_nd_cuda);
    m.impl("circular_pad_nd_backward", circular_pad_nd_backward_cuda);
    m.impl("constant_pad_nd", constant_pad_nd_cuda);
    m.impl("constant_pad_nd_backward", constant_pad_nd_backward_cuda);
    // The rank-specific spellings are the same kernel: the pad list length
    // selects the padded trailing dimensions, so 1d/2d/3d collapse onto _nd.
    m.impl("reflection_pad1d", reflection_pad_nd_cuda);
    m.impl("reflection_pad2d", reflection_pad_nd_cuda);
    m.impl("reflection_pad3d", reflection_pad_nd_cuda);
    m.impl("reflection_pad1d_backward", reflection_pad_nd_backward_cuda);
    m.impl("reflection_pad2d_backward", reflection_pad_nd_backward_cuda);
    m.impl("reflection_pad3d_backward", reflection_pad_nd_backward_cuda);
    m.impl("replication_pad1d", replication_pad_nd_cuda);
    m.impl("replication_pad2d", replication_pad_nd_cuda);
    m.impl("replication_pad3d", replication_pad_nd_cuda);
    m.impl("replication_pad1d_backward", replication_pad_nd_backward_cuda);
    m.impl("replication_pad2d_backward", replication_pad_nd_backward_cuda);
    m.impl("replication_pad3d_backward", replication_pad_nd_backward_cuda);

    m.impl("reflection_pad1d_backward.grad_input", reflection_pad1d_backward_grad_input_cuda);
    m.impl("reflection_pad2d_backward.grad_input", reflection_pad2d_backward_grad_input_cuda);
    m.impl("reflection_pad3d_backward.grad_input", reflection_pad3d_backward_grad_input_cuda);
    m.impl("replication_pad1d_backward.grad_input", replication_pad1d_backward_grad_input_cuda);
    m.impl("replication_pad2d_backward.grad_input", replication_pad2d_backward_grad_input_cuda);
    m.impl("replication_pad3d_backward.grad_input", replication_pad3d_backward_grad_input_cuda);
}

}  // namespace cuda
}  // namespace tensorplay
