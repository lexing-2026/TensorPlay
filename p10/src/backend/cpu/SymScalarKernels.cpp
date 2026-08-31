#include "Tensor.h"

#include <cstdint>

#include "Dispatcher.h"
#include "MemoryFormat.h"

namespace tensorplay {
namespace cpu {

SymInt sym_size_cpu(const Tensor& self, int64_t dim) {
    return SymInt(self.size(dim));
}

SymBool sym_is_contiguous_cpu(const Tensor& self, int64_t memory_format) {
    return SymBool(self.is_contiguous(
        static_cast<MemoryFormat>(memory_format)));
}

SymInt sym_numel_cpu(const Tensor& self) {
    return SymInt(self.numel());
}

SymInt sym_storage_offset_cpu(const Tensor& self) {
    if (!self.defined()) {
        return SymInt(0);
    }
    return SymInt(static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset()));
}

SymInt sym_stride_cpu(const Tensor& self, int64_t dim) {
    return SymInt(self.stride(dim));
}

}  // namespace cpu

namespace composite {

SymInt sym_size_composite(const Tensor& self, int64_t dim) {
    return SymInt(self.size(dim));
}

SymBool sym_is_contiguous_composite(
    const Tensor& self, int64_t memory_format) {
    return SymBool(self.is_contiguous(
        static_cast<MemoryFormat>(memory_format)));
}

SymInt sym_numel_composite(const Tensor& self) {
    return SymInt(self.numel());
}

SymInt sym_storage_offset_composite(const Tensor& self) {
    if (!self.defined()) {
        return SymInt(0);
    }
    return SymInt(static_cast<int64_t>(
        self.unsafeGetTensorImpl()->storage_offset()));
}

SymInt sym_stride_composite(const Tensor& self, int64_t dim) {
    return SymInt(self.stride(dim));
}

}  // namespace composite

TENSORPLAY_LIBRARY_IMPL(CPU, SymScalarKernels) {
    m.impl("sym_size.int", cpu::sym_size_cpu);
    m.impl("sym_is_contiguous", cpu::sym_is_contiguous_cpu);
    m.impl("sym_numel", cpu::sym_numel_cpu);
    m.impl("sym_storage_offset", cpu::sym_storage_offset_cpu);
    m.impl("sym_stride.int", cpu::sym_stride_cpu);
}

TENSORPLAY_LIBRARY_IMPL(Composite, SymScalarComposites) {
    m.impl("sym_size.int", composite::sym_size_composite);
    m.impl("sym_is_contiguous", composite::sym_is_contiguous_composite);
    m.impl("sym_numel", composite::sym_numel_composite);
    m.impl("sym_storage_offset", composite::sym_storage_offset_composite);
    m.impl("sym_stride.int", composite::sym_stride_composite);
}

}  // namespace tensorplay
