// Tensor-shape aliases.

#include "SetStorage.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay::cuda {

namespace ops = tensorplay::tpx::ops;

Tensor concat_native_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor concatenate_native_cuda(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor diagflat_native_cuda(const Tensor& self, int64_t offset) {
    Tensor flat = ops::view(ops::contiguous(self, 0), {-1});
    return ops::diag(flat, offset);
}

Tensor& set_tensor_cuda(Tensor& self, const Tensor& source) {
    return native::set_tensor_native(self, source);
}

Tensor& set_storage_cuda(Tensor& self, Storage source) {
    return native::set_storage_native(self, std::move(source));
}

Tensor& set_storage_cuda_(Tensor& self, Storage source, int64_t storage_offset,
                          const std::vector<int64_t>& size,
                          const std::vector<int64_t>& stride) {
    return native::set_storage_offset_native(
        self, std::move(source), storage_offset, size, stride);
}

Tensor& set_cuda_(Tensor& self) {
    return native::reset_tensor_storage_native(self, DeviceType::CUDA);
}

TENSORPLAY_LIBRARY_IMPL(CUDA, NativeTensorShape) {
    m.impl("concat", concat_native_cuda);
    m.impl("concatenate", concatenate_native_cuda);
    m.impl("diagflat", diagflat_native_cuda);
    m.impl("set_.source_Storage", set_storage_cuda);
    m.impl("set_.source_Storage_storage_offset", set_storage_cuda_);
    m.impl("set_.source_Tensor", set_tensor_cuda);
    m.impl("set_", set_cuda_);
}

} // namespace tensorplay::cuda
