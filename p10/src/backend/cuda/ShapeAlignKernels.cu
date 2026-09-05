// batch (see cpu/ShapeAlignKernels.cpp for the composite commentary).
//
// The device-generic composites of this batch are registered once under the
// backend-neutral Composite key (p10/src/RegisterComposites.cpp) and are
// served to CUDA tensors by the
// dispatcher's composite fallthrough.  The only op with real per-device code
// is repeat(): a single-pass index-math gather implemented here as a
// per-backend override. Other shape operations use the composite fallthrough.

#include "Tensor.h"
#include "CUDARuntime.h"
#include "Exception.h"
#include "Quantizer.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

int64_t checked_repeat_extent(int64_t source, int64_t repeat) {
    if (repeat < 0) {
        TP_THROW(RuntimeError, "repeat: repeats must be non-negative");
    }
    if (source == 0 || repeat == 0) return 0;
    if (source > std::numeric_limits<int64_t>::max() / repeat) {
        TP_THROW(RuntimeError, "repeat: resulting dimension is too large");
    }
    return source * repeat;
}

void check_repeat_numel(const std::vector<int64_t>& target) {
    int64_t total = 1;
    for (const int64_t extent : target) {
        if (extent == 0) return;
        if (total > std::numeric_limits<int64_t>::max() / extent) {
            TP_THROW(RuntimeError, "repeat: resulting tensor is too large");
        }
        total *= extent;
    }
}

Tensor make_repeat_output(const Tensor& self,
                          const std::vector<int64_t>& target) {
    if (!isQuantizedType(self.dtype())) {
        return Tensor::empty(target, self.dtype(), self.device());
    }
    quantized::require_quantized(self, "repeat");
    Tensor codes = Tensor::empty(target, underlying_storage_type(self.dtype()),
                                 self.device());
    return quantized::make_qtensor(codes, quantized::quantizer_of(self),
                                   self.dtype());
}

#define CUDA_CHECK(condition)                                                        \
    do {                                                                             \
        cudaError_t error = (condition);                                             \
        if (error != cudaSuccess) {                                                  \
            TP_THROW(RuntimeError, "CUDA error: ", cudaGetErrorString(error),        \
                     " at ", __FILE__, ":", __LINE__);                               \
        }                                                                            \
    } while (0)

template <typename T>
__global__ void repeat_gather_kernel(int64_t n, int64_t out_nd,
                                     const int64_t* __restrict__ tstrides,
                                     const int64_t* __restrict__ padded,
                                     const int64_t* __restrict__ pstrides,
                                     const T* __restrict__ src,
                                     T* __restrict__ dst) {
    const int64_t grid_stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t f = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         f < n; f += grid_stride) {
        int64_t rem = f, off = 0;
        for (int64_t i = 0; i < out_nd; ++i) {
            const int64_t c = rem / tstrides[i];
            rem %= tstrides[i];
            off += (c % padded[i]) * pstrides[i];
        }
        dst[f] = src[off];
    }
}

} // anonymous namespace

Tensor repeat_cuda(const Tensor& self, const std::vector<int64_t>& repeats) {
    const int64_t nd = self.dim();
    if (static_cast<int64_t>(repeats.size()) < nd) {
        TP_THROW(RuntimeError,
                 "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");
    }

    const int64_t out_nd = static_cast<int64_t>(repeats.size());
    Tensor sc = self.contiguous();
    std::vector<int64_t> padded(out_nd, 1), padded_strides(out_nd, 0), target(out_nd);
    for (int64_t i = 0; i < nd; ++i) {
        padded[out_nd - nd + i] = sc.size(i);
        padded_strides[out_nd - nd + i] = sc.stride(i);
    }
    bool zero = false;
    for (int64_t i = 0; i < out_nd; ++i) {
        zero = zero || repeats[i] == 0;
        target[i] = checked_repeat_extent(padded[i], repeats[i]);
    }
    check_repeat_numel(target);

    OptionalCUDAGuard device_guard(self.device());
    Tensor out = make_repeat_output(self, target);
    const int64_t total = out.numel();
    if (zero || total == 0) return out;

    std::vector<int64_t> tstrides(out_nd, 1);
    for (int64_t i = out_nd - 2; i >= 0; --i) tstrides[i] = tstrides[i + 1] * target[i + 1];

    // Stage the small host-side index arrays on-device.
    std::vector<int64_t> h_idx;
    h_idx.reserve(static_cast<size_t>(out_nd) * 3);
    h_idx.insert(h_idx.end(), tstrides.begin(), tstrides.end());
    h_idx.insert(h_idx.end(), padded.begin(), padded.end());
    h_idx.insert(h_idx.end(), padded_strides.begin(), padded_strides.end());
    Tensor idx_d = Tensor::empty({static_cast<int64_t>(h_idx.size())}, DType::Int64,
                                 self.device());
    CUDA_CHECK(cudaMemcpy(idx_d.data_ptr<int64_t>(), h_idx.data(),
                          h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    const int64_t* tstrides_d = idx_d.data_ptr<int64_t>();
    const int64_t* padded_d = tstrides_d + out_nd;
    const int64_t* pstrides_d = padded_d + out_nd;

    const int blocks = static_cast<int>(
        std::min<int64_t>((total + kThreads - 1) / kThreads, kMaxBlocks));
    auto stream = getCurrentCUDAStream().stream();

#define TP_REPEAT_CUDA_CASE(ctype, name)                                                \
    case DType::name:                                                                   \
        repeat_gather_kernel<ctype><<<blocks, kThreads, 0, stream>>>(                   \
            total, out_nd, tstrides_d, padded_d, pstrides_d,                            \
            sc.data_ptr<ctype>(), out.data_ptr<ctype>());                               \
        break;

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_REPEAT_CUDA_CASE)
        TENSORPLAY_FORALL_QINT_TYPES(TP_REPEAT_CUDA_CASE)
        default: TP_THROW(TypeError, "repeat: unsupported dtype");
    }
#undef TP_REPEAT_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Composite shapeops kernels (expand/stack/split/atleast/fill/equal/allclose
// families) are registered ONCE under the backend-neutral Composite key from
// p10/src/RegisterComposites.cpp -- TensorPlay's composite registration
// serves them to every dense backend until overridden.  This TU only carries
// repeat(), whose gather is real device code.
} // namespace cuda

namespace cuda {

TENSORPLAY_LIBRARY_IMPL(CUDA, ShapeAlign) {
    m.impl("repeat", repeat_cuda);
}

} // namespace cuda
} // namespace tensorplay
