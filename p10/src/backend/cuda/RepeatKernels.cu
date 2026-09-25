#include "CudaDispatchHelpers.cuh"

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

Tensor repeat_interleave_tensor_cuda(const Tensor& repeats,
                                     std::optional<int64_t> output_size) {
    TP_CHECK(repeats.dim() == 1,
             "repeat_interleave: repeats must be 1-dimensional");
    TP_CHECK(repeats.dtype() == DType::Int32 ||
                 repeats.dtype() == DType::Int64,
             "repeats must have Int32 or Int64 dtype");
    if (repeats.numel() == 0) {
        return Tensor::empty({0}, repeats.dtype(), repeats.device());
    }

    Tensor rep = repeats.contiguous();
    Tensor ends = ops::cumsum(rep, 0, DType::Int64);
    const int64_t required_size =
        ends.select(0, ends.size(0) - 1).item<int64_t>();
    const int64_t total = output_size.value_or(required_size);
    TP_CHECK(total == required_size,
             "allocated size does not match required size");
    TP_CHECK(rep.ge(Scalar(0)).all().item<bool>(),
             "repeats can not be negative");

    Tensor ar = ops::arange(Scalar(int64_t(0)), Scalar(total),
                            Scalar(int64_t(1)), DType::Int64, repeats.device());
    Tensor indices = ops::searchsorted(ends, ar, false, true);
    return repeats.dtype() == DType::Int32
        ? indices.to(DType::Int32)
        : indices;
}

TENSORPLAY_LIBRARY_IMPL(CUDA, RepeatKernels) {
    // misc internal spellings
    m.impl("repeat_interleave.Tensor", interop_repeat_interleave_Tensor_cuda);
}

} // namespace cuda
} // namespace tensorplay
