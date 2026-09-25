#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


Tensor tri_diag_base(int64_t row, int64_t col, int64_t offset, bool lower,
                     const Device& dev) {
    // Row r contributes clamp(r + offset + 1, 0, col) cells below the
    // diagonal, clamp(col - max(r + offset, 0), 0, col) cells above it.
    Tensor lead = ops::add(ops::arange(Scalar(int64_t(0)), Scalar(row),
                                       Scalar(int64_t(1)), DType::Int64, dev),
                          Scalar(offset));
    Tensor counts = lower
        ? ops::clamp(ops::add(lead, Scalar(int64_t(1))),
                     Scalar(int64_t(0)), Scalar(col))
        : ops::clamp(Scalar(col) - ops::clamp(lead, Scalar(int64_t(0)),
                                              Scalar(col)),
                     Scalar(int64_t(0)), Scalar(col));
    return counts;
}


Tensor interop_tril_indices_cuda(int64_t row, int64_t col, int64_t offset,
                                 DType dtype, std::optional<Device> device,
                                 bool pin_memory) {
    const Device dev = device.value_or(Device(DeviceType::CUDA));
    Tensor counts = tri_diag_base(row, col, offset, /*lower=*/true, dev);
    const int64_t count =
        static_cast<int64_t>(counts.sum().item().to<int64_t>());
    Tensor rows = repeat_interleave_tensor_cuda(counts, count);
    Tensor starts = ops::sub(ops::cumsum(counts, 0), counts);
    Tensor flat = ops::arange(Scalar(int64_t(0)), Scalar(count),
                              Scalar(int64_t(1)), DType::Int64, dev);
    Tensor cols = ops::sub(flat, ops::index_select(starts, 0, rows));
    Tensor result = ops::empty({2, count}, dtype, dev, pin_memory);
    result.select(0, 0).copy_(rows.to(dtype));
    result.select(0, 1).copy_(cols.to(dtype));
    return result;
}


Tensor interop_triu_indices_cuda(int64_t row, int64_t col, int64_t offset,
                                 DType dtype, std::optional<Device> device,
                                 bool pin_memory) {
    const Device dev = device.value_or(Device(DeviceType::CUDA));
    Tensor counts = tri_diag_base(row, col, offset, /*lower=*/false, dev);
    const int64_t count =
        static_cast<int64_t>(counts.sum().item().to<int64_t>());
    Tensor rows = repeat_interleave_tensor_cuda(counts, count);
    Tensor starts = ops::sub(ops::cumsum(counts, 0), counts);
    Tensor flat = ops::arange(Scalar(int64_t(0)), Scalar(count),
                              Scalar(int64_t(1)), DType::Int64, dev);
    // The first cell of row r sits at column max(r + offset, 0).
    Tensor first_col = ops::clamp(
        ops::add(ops::arange(Scalar(int64_t(0)), Scalar(row),
                             Scalar(int64_t(1)), DType::Int64, dev),
                 Scalar(offset)),
        Scalar(int64_t(0)), Scalar(col));
    Tensor cols = ops::add(ops::index_select(first_col, 0, rows),
                           ops::sub(flat, ops::index_select(starts, 0, rows)));
    Tensor result = ops::empty({2, count}, dtype, dev, pin_memory);
    result.select(0, 0).copy_(rows.to(dtype));
    result.select(0, 1).copy_(cols.to(dtype));
    return result;
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, TensorFactoriesKernels) {
    m.impl("tril_indices", interop_tril_indices_cuda);
    m.impl("triu_indices", interop_triu_indices_cuda);
}

} // namespace cuda
} // namespace tensorplay
