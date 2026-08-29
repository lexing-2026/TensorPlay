// batch (see cpu/ShapeAlignKernels.cpp for the composite commentary).
//
// The device-generic composites of this batch are registered once under the
// backend-neutral Composite key (p10/src/RegisterComposites.cpp, upstream's
// CompositeExplicitAutograd mapping) and are served to CUDA tensors by the
// dispatcher's composite fallthrough.  The only op with real per-device code
// is repeat(): a single-pass index-math gather mirroring upstream's
// here as a per-backend override -- the same pattern upstream uses for MPS
// (native_functions.yaml: MPS: repeat_mps).

#include "Tensor.h"
#include "CUDARuntime.h"
#include "Exception.h"

#include <algorithm>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

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
        target[i] = padded[i] * repeats[i];
    }
    // Negative repeats surface through the output allocation exactly like
    for (const int64_t x : target) {
        if (x < 0) {
            std::string sizes = "[";
            for (size_t i = 0; i < target.size(); ++i) {
                if (i) sizes += ", ";
                sizes += std::to_string(target[i]);
            }
            sizes += "]";
            TP_THROW(RuntimeError, "Trying to create tensor with negative dimension ",
                     x, ": ", sizes);
        }
    }

    OptionalCUDAGuard device_guard(self.device());
    Tensor out = Tensor::empty(target, self.dtype(), self.device());
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
        default: TP_THROW(TypeError, "repeat: unsupported dtype");
    }
#undef TP_REPEAT_CUDA_CASE
    CUDA_CHECK(cudaGetLastError());
    return out;
}

// Composite shapeops kernels (expand/stack/split/atleast/fill/equal/allclose
// families) are registered ONCE under the backend-neutral Composite key from
// p10/src/RegisterComposites.cpp -- TensorPlay's analog of the generated
// serves them to every dense backend until overridden.  This TU only carries
// repeat(), whose gather is real device code -- the same per-backend override
// pattern upstream uses for MPS (native_functions.yaml: MPS: repeat_mps).
} // namespace cuda

namespace cuda {

TENSORPLAY_LIBRARY_IMPL(CUDA, ShapeAlign) {
    m.impl("repeat", repeat_cuda);
}

} // namespace cuda
} // namespace tensorplay
