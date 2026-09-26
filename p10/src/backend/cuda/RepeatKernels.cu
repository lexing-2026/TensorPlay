#include "CudaDispatchHelpers.cuh"
#include "CUDARuntime.h"

namespace tensorplay {
namespace cuda {


namespace {


Tensor interop_repeat_interleave_Tensor_cuda(
        const Tensor& repeats, std::optional<int64_t> output_size) {
    return repeat_interleave_tensor_cuda(repeats, output_size);
}

} // namespace


// ---------------------------------------------------------------------------
// repeat_interleave.Tensor returns the flat source-index list
// [0 x r0, 1 x r1, ...] from cumulative repeat boundaries.
// ---------------------------------------------------------------------------

namespace {

// One warp per entry of `repeats`: the warp walks the run that entry expands to
// and writes the source index straight into the result.  Deriving the run for
// every output element instead (a search per element over the cumulative
// boundaries) would read the whole boundary table log(size) times and need a
// materialized index vector.
template <typename index_t>
__global__ void repeat_interleave_fill_kernel(const index_t* repeat_ptr,
                                              const int64_t* cumsum_ptr,
                                              index_t* result, int64_t size,
                                              int64_t result_size) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t warp_count = (static_cast<int64_t>(blockDim.x) * gridDim.x) / 32;
    const int64_t warp_id = idx / 32;
    const int64_t lane = idx % 32;
    for (int64_t i = warp_id; i < size; i += warp_count) {
        const int64_t end = cumsum_ptr[i];
        const int64_t repeat = repeat_ptr[i];
        const int64_t start = end - repeat;
        for (int64_t j = start + lane; j < end; j += 32) {
            result[j] = static_cast<index_t>(i);
        }
    }
}

template <typename index_t>
void launch_repeat_interleave_fill(const Tensor& rep, const Tensor& ends,
                                   Tensor& result, int64_t size,
                                   int64_t result_size) {
    constexpr int64_t kBlock = 512;
    const int64_t warps_per_block = kBlock / 32;
    const int64_t grid = std::min<int64_t>((size + warps_per_block - 1) /
                                                warps_per_block,
                                            2048);
    repeat_interleave_fill_kernel<index_t>
        <<<static_cast<unsigned>(grid), static_cast<unsigned>(kBlock), 0,
           getCurrentCUDAStream().stream()>>>(
            rep.data_ptr<index_t>(), ends.data_ptr<int64_t>(),
            result.data_ptr<index_t>(), size, result_size);
    checkCuda(cudaGetLastError(), "repeat_interleave fill kernel");
}

}  // namespace

Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size) {
    TP_CHECK(repeats.dim() == 1,
             "repeat_interleave: repeats must be 1-dimensional");
    // The index widths the fill kernel is instantiated for; narrower repeats
    // are accepted because the run boundaries are what index the result.
    TP_CHECK(repeats.dtype() == DType::Int8 ||
                 repeats.dtype() == DType::UInt8 ||
                 repeats.dtype() == DType::Int16 ||
                 repeats.dtype() == DType::Int32 ||
                 repeats.dtype() == DType::Int64,
             "repeats must have an 8/16/32/64-bit integer dtype");
    if (repeats.numel() == 0) {
        return Tensor::empty({0}, repeats.dtype(), repeats.device());
    }

    // The cumulative boundaries are computed in 64 bits, so a narrow repeats
    // tensor is widened once rather than needing its own scan.
    Tensor rep = repeats.contiguous();
    if (rep.dtype() != DType::Int32 && rep.dtype() != DType::Int64) {
        rep = rep.to(DType::Int32);
    }
    const int64_t size = rep.numel();
    Tensor ends = ops::cumsum(rep, 0, DType::Int64);
    int64_t total = 0;
    if (output_size.has_value()) {
        // The caller states the size, so the boundaries are never read back;
        // the fill kernel is what notices a mismatch.
        total = output_size.value();
    } else {
        total = ends.select(0, size - 1).item<int64_t>();
        TP_CHECK(rep.ge(Scalar(0)).all().item<bool>(),
                 "repeats can not be negative");
    }

    Tensor result = Tensor::empty({total}, repeats.dtype(), repeats.device());
    if (total == 0) return result;
    switch (rep.dtype()) {
        case DType::Int32:
            launch_repeat_interleave_fill<int32_t>(rep, ends, result, size, total);
            break;
        default:
            launch_repeat_interleave_fill<int64_t>(rep, ends, result, size, total);
            break;
    }
    return result;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RepeatKernels) {
    // misc internal spellings
    m.impl("repeat_interleave.Tensor", interop_repeat_interleave_Tensor_cuda);
}

} // namespace cuda
} // namespace tensorplay
