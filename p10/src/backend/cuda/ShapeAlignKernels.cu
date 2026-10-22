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
#include "CUDALoops.cuh"
#include "Exception.h"
#include "Quantizer.h"

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace tensorplay {
namespace cuda {

namespace {

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

// One block copies one chunk of one contiguous source run into the place
// that run takes in the answer.  The run is a whole number of elements that
// the innermost axis repeats, so its bytes move with no index arithmetic
// inside the copy; the source offset is resolved once per block.
//
// When a single outer axis repeats the source, the blocks that write the
// replicas of one chunk are dispatched next to each other, so the run is
// fetched from memory once and the neighbouring blocks read it from cache.
// The replica index is the fastest part of the block index for that reason.
template <typename VecT>
__global__ void repeat_run_copy_kernel(
    const char* __restrict__ src, char* __restrict__ dst,
    int64_t run, int64_t runs, int64_t chunks_per_run,
    int64_t chunk_bytes, int64_t out_nd,
    const int64_t* __restrict__ tstrides, const int64_t* __restrict__ padded,
    const int64_t* __restrict__ pstrides, int64_t group, int64_t item,
    int64_t outer_reps, int64_t outer_stride) {
    const int64_t run_bytes = run * item;
    const int64_t per_group = chunks_per_run * runs;
    const int64_t src_block = outer_reps > 1 ? blockIdx.x / outer_reps : blockIdx.x;
    const int64_t replica =
        outer_reps > 1 ? blockIdx.x - src_block * outer_reps : 0;
    const int64_t g = src_block / per_group;
    const int64_t rest = src_block - g * per_group;
    const int64_t r = rest / chunks_per_run;
    const int64_t ck = rest - r * chunks_per_run;

    int64_t off;
    int64_t base;
    if (outer_reps > 1) {
        // `g` names a source run, and the replica sits `outer_stride` groups
        // past the one the source coordinates put it at.
        off = g * run;
        base = 0;
        int64_t rem = g;
        for (int64_t i = out_nd - 2; i >= 0; --i) {
            const int64_t c = rem % padded[i];
            rem /= padded[i];
            base += c * (tstrides[i] / group);
        }
        base += replica * outer_stride;
    } else {
        int64_t rem = g * group;
        off = 0;
        for (int64_t i = 0; i < out_nd - 1; ++i) {
            const int64_t c = rem / tstrides[i];
            rem -= c * tstrides[i];
            off += (c % padded[i]) * pstrides[i];
        }
        base = g;
    }

    const int64_t step = ck * chunk_bytes;
    const char* s = src + off * item + step;
    char* d = dst + (base * group + r * run) * item + step;
    const int64_t remaining = run_bytes - step;
    const int64_t nbytes = remaining < chunk_bytes ? remaining : chunk_bytes;

    const int64_t nv = nbytes / static_cast<int64_t>(sizeof(VecT));
    const VecT* sv = reinterpret_cast<const VecT*>(s);
    VecT* dv = reinterpret_cast<VecT*>(d);
    for (int64_t j = threadIdx.x; j < nv; j += blockDim.x) dv[j] = sv[j];
    if constexpr (sizeof(VecT) > 1) {
        for (int64_t j = nv * static_cast<int64_t>(sizeof(VecT)) + threadIdx.x;
             j < nbytes; j += blockDim.x) {
            d[j] = s[j];
        }
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

    // A run is the widest stretch the innermost axis contributes, and the
    // axes above it only choose which run goes where.  Moving whole runs
    // lets one block copy a chunk with no per-element index arithmetic, and
    // the chunk offsets stay aligned to the widest word the run allows.
    const int64_t run = padded[out_nd - 1];
    const int64_t runs = run > 0 ? target[out_nd - 1] / run : 0;
    const int64_t group = run * runs;
    const int64_t groups = group > 0 ? total / group : 0;
    const size_t item = self.itemsize();
    const int64_t run_bytes = static_cast<int64_t>(run) * static_cast<int64_t>(item);
    constexpr int64_t kChunkBytes = 1 << 16;
    const int64_t chunks_per_run =
        run_bytes > 0 ? (run_bytes + kChunkBytes - 1) / kChunkBytes : 0;

    // A single repeated outer axis lets the blocks that write its replicas be
    // dispatched together, so the fetched run is shared through the cache
    // instead of being read once per replica.  The ordering is kept for small
    // replica counts and shapes with enough source blocks to fill the device.
    int64_t outer_reps = 1;
    int64_t outer_stride = 0;
    {
        int64_t rep_axis = -1;
        int64_t rep_count = 1;
        bool single_axis = true;
        for (int64_t i = 0; i < out_nd - 1; ++i) {
            const int64_t f = padded[i] > 0 ? target[i] / padded[i] : 0;
            if (f > 1) {
                if (rep_axis >= 0) {
                    single_axis = false;
                    break;
                }
                rep_axis = i;
                rep_count = f;
            }
        }
        const int64_t run_blocks =
            (groups / rep_count) * runs * chunks_per_run;
        if (single_axis && rep_axis >= 0 && rep_count <= 16 &&
            groups % rep_count == 0 && run_bytes >= 4096 &&
            run_blocks >= 1024) {
            outer_reps = rep_count;
            outer_stride = (padded[rep_axis] * tstrides[rep_axis]) / group;
        }
    }
    const int64_t blocks = groups * runs * chunks_per_run;
    if (!isQuantizedType(self.dtype()) && run > 0 && blocks > 0 &&
        blocks <= std::numeric_limits<int32_t>::max()) {
        const char* src_bytes = static_cast<const char*>(sc.const_data_ptr());
        char* dst_bytes = static_cast<char*>(out.data_ptr());
        const uintptr_t sp = reinterpret_cast<uintptr_t>(src_bytes);
        const uintptr_t dp = reinterpret_cast<uintptr_t>(dst_bytes);
        const int kThreads = 256;
        auto stream = getCurrentCUDAStream().stream();
        const unsigned grid = static_cast<unsigned>(blocks);
        if (run_bytes % 16 == 0 && sp % 16 == 0 && dp % 16 == 0) {
            repeat_run_copy_kernel<uint4><<<grid, kThreads, 0, stream>>>(
                src_bytes, dst_bytes, run, runs, chunks_per_run, kChunkBytes,
                out_nd, tstrides_d, padded_d, pstrides_d, group,
                static_cast<int64_t>(item), outer_reps, outer_stride);
        } else if (run_bytes % 8 == 0 && sp % 8 == 0 && dp % 8 == 0) {
            repeat_run_copy_kernel<uint2><<<grid, kThreads, 0, stream>>>(
                src_bytes, dst_bytes, run, runs, chunks_per_run, kChunkBytes,
                out_nd, tstrides_d, padded_d, pstrides_d, group,
                static_cast<int64_t>(item), outer_reps, outer_stride);
        } else if (run_bytes % 4 == 0 && sp % 4 == 0 && dp % 4 == 0) {
            repeat_run_copy_kernel<unsigned int><<<grid, kThreads, 0, stream>>>(
                src_bytes, dst_bytes, run, runs, chunks_per_run, kChunkBytes,
                out_nd, tstrides_d, padded_d, pstrides_d, group,
                static_cast<int64_t>(item), outer_reps, outer_stride);
        } else {
            repeat_run_copy_kernel<unsigned char><<<grid, kThreads, 0, stream>>>(
                src_bytes, dst_bytes, run, runs, chunks_per_run, kChunkBytes,
                out_nd, tstrides_d, padded_d, pstrides_d, group,
                static_cast<int64_t>(item), outer_reps, outer_stride);
        }
        loop_launch_check("repeat_run_copy_kernel");
        return out;
    }

#define TP_REPEAT_CUDA_CASE(ctype, name)                                                \
    case DType::name: {                                                                  \
        const ctype* source = sc.data_ptr<ctype>();                                      \
        gpu_kernel_with_index(out, [=] GPU_LAMBDA(int64_t flat_index) -> ctype {        \
            int64_t remainder = flat_index;                                              \
            int64_t source_offset = 0;                                                   \
            for (int64_t i = 0; i < out_nd; ++i) {                                       \
                const int64_t coordinate = remainder / tstrides_d[i];                    \
                remainder %= tstrides_d[i];                                              \
                source_offset += (coordinate % padded_d[i]) * pstrides_d[i];              \
            }                                                                             \
            return source[source_offset];                                                \
        });                                                                               \
        break;                                                                            \
    }

    switch (self.dtype()) {
        TENSORPLAY_FORALL_SCALAR_TYPES(TP_REPEAT_CUDA_CASE)
        TENSORPLAY_FORALL_QINT_TYPES(TP_REPEAT_CUDA_CASE)
        default: TP_THROW(TypeError, "repeat: unsupported dtype");
    }
#undef TP_REPEAT_CUDA_CASE
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
