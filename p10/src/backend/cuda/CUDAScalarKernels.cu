#include "CudaDispatchHelpers.cuh"

namespace tensorplay {
namespace cuda {


namespace {


// ---------------------------------------------------------------------------
// _local_scalar_dense: read a 0-dim value off the device into a Scalar.
// ---------------------------------------------------------------------------

Scalar interop__local_scalar_dense_cuda(const Tensor& self) {
    TP_CHECK(self.numel() == 1,
             "_local_scalar_dense only supports 1-element tensors, got ",
             self.numel());
    Tensor host = self.reshape({1}).contiguous().to(Device(DeviceType::CPU));
    switch (self.dtype()) {
        case DType::Float32: return Scalar(host.data_ptr<float>()[0]);
        case DType::Float64: return Scalar(host.data_ptr<double>()[0]);
        case DType::Int64: return Scalar(host.data_ptr<int64_t>()[0]);
        case DType::Int32: return Scalar(static_cast<int64_t>(host.data_ptr<int32_t>()[0]));
        case DType::Float16: return Scalar(static_cast<double>(host.data_ptr<Half>()[0]));
        case DType::BFloat16: return Scalar(static_cast<double>(host.data_ptr<BFloat16>()[0]));
        case DType::Bool: return Scalar(host.data_ptr<bool>()[0]);
        case DType::Int16: return Scalar(static_cast<int64_t>(host.data_ptr<int16_t>()[0]));
        case DType::Int8: return Scalar(static_cast<int64_t>(host.data_ptr<int8_t>()[0]));
        case DType::UInt8: return Scalar(static_cast<int64_t>(host.data_ptr<uint8_t>()[0]));
        default:
            TP_THROW(TypeError, "_local_scalar_dense: unsupported dtype ",
                     toString(self.dtype()));
    }
}

} // namespace

TENSORPLAY_LIBRARY_IMPL(CUDA, CUDAScalarKernels) {
    m.impl("_local_scalar_dense", interop__local_scalar_dense_cuda);
}

} // namespace cuda
} // namespace tensorplay
