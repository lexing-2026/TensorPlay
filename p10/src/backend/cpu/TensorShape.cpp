// Tensor-shape aliases.

#include "SetStorage.h"
#include "Tensor.h"
#include "Dispatcher.h"
#include "tensorplay/ops/TPXOpsGenerated.h"

#include <vector>

namespace tensorplay::cpu {

namespace ops = tensorplay::tpx::ops;

Tensor concat_native_cpu(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor concatenate_native_cpu(const std::vector<Tensor>& tensors, int64_t dim) {
    return ops::cat(tensors, dim);
}

Tensor diagflat_native_cpu(const Tensor& self, int64_t offset) {
    Tensor flat = ops::view(ops::contiguous(self, 0), {-1});
    return ops::diag(flat, offset);
}

Tensor& set_tensor_cpu(Tensor& self, const Tensor& source) {
    return native::set_tensor_native(self, source);
}

Tensor& set_storage_cpu(Tensor& self, Storage source) {
    return native::set_storage_native(self, std::move(source));
}

Tensor& set_storage_cpu_(Tensor& self, Storage source, int64_t storage_offset,
                         const std::vector<int64_t>& size,
                         const std::vector<int64_t>& stride) {
    return native::set_storage_offset_native(
        self, std::move(source), storage_offset, size, stride);
}

Tensor& set_cpu_(Tensor& self) {
    return native::reset_tensor_storage_native(self, DeviceType::CPU);
}

TENSORPLAY_LIBRARY_IMPL(CPU, NativeTensorShape) {
    m.impl("concat", concat_native_cpu);
    m.impl("concatenate", concatenate_native_cpu);
    m.impl("diagflat", diagflat_native_cpu);
    m.impl("set_.source_Storage", set_storage_cpu);
    m.impl("set_.source_Storage_storage_offset", set_storage_cpu_);
    m.impl("set_.source_Tensor", set_tensor_cpu);
    m.impl("set_", set_cpu_);
}

} // namespace tensorplay::cpu
