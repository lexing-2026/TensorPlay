// Allocation, copies, casts and fills: the kernels every other kernel of the
// backend (and the host fallback) is built on.

#include <cstring>
#include <optional>
#include <utility>

#include "AscendAllocator.h"
#include "AscendOpApi.h"
#include "AscendRuntime.h"
#include "AscendTensor.h"
#include "Dispatcher.h"
#include "LocalDispatchKeySet.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace tensorplay {
namespace npu {

namespace {

int64_t numel_of(const std::vector<int64_t>& sizes) {
    int64_t n = 1;
    for (int64_t d : sizes) {
        TP_CHECK(d >= 0, "NPU: negative dimension ", d, " in a requested size");
        n *= d;
    }
    return n;
}

Tensor host_tensor(const std::vector<int64_t>& sizes, DType dtype) {
    return Tensor(Storage(static_cast<size_t>(numel_of(sizes)) * elementSize(dtype)), sizes,
                  dtype);
}

Tensor to_host_dense(const Tensor& src) {
    const Tensor dense = contiguous_on_device(src);
    Tensor host = host_tensor(dense.sizes().vec(), dense.dtype());
    copy_to_host(host.data_ptr(), dense.data_ptr(),
                 static_cast<size_t>(dense.numel()) * elementSize(dense.dtype()),
                 index_of(dense.device()));
    return host;
}

void host_to_device(Tensor& self, const Tensor& src) {
    // Shape and dtype are settled on the host, so only dense bytes move.
    Tensor host = src;
    if (host.sizes().vec() != self.sizes().vec()) host = host.expand(self.sizes().vec());
    if (host.dtype() != self.dtype()) host = host.to(self.dtype());
    host = host.contiguous();
    const size_t nbytes = static_cast<size_t>(self.numel()) * elementSize(self.dtype());
    const int index = index_of(self.device());
    if (self.is_contiguous()) {
        copy_to_device(self.data_ptr(), host.data_ptr(), nbytes, index);
        return;
    }
    Tensor staging = empty_on(self.sizes().vec(), self.dtype(), index);
    copy_to_device(staging.data_ptr(), host.data_ptr(), nbytes, index);
    strided_copy(self, staging);
}

void device_to_host(Tensor& self, const Tensor& src) {
    if (self.is_contiguous() && src.is_contiguous() && self.dtype() == src.dtype() &&
        self.sizes().vec() == src.sizes().vec()) {
        copy_to_host(self.data_ptr(), src.data_ptr(),
                     static_cast<size_t>(self.numel()) * elementSize(self.dtype()),
                     index_of(src.device()));
        return;
    }
    host_copy_into(self, to_host_dense(src));
}

void device_to_device(Tensor& self, const Tensor& src) {
    const int index = index_of(self.device());
    if (index_of(src.device()) != index) {
        host_to_device(self, to_host_dense(src));
        return;
    }
    Tensor source = src;
    if (source.sizes().vec() != self.sizes().vec()) source = source.expand(self.sizes().vec());
    if (source.dtype() == self.dtype()) {
        strided_copy(self, source);
        return;
    }
    // The operator library converts the values on the device; the result
    // then takes the ordinary strided copy into ``self``.
    if (!source.is_contiguous()) source = contiguous_on_device(source);
    strided_copy(self, cast_on_device(source, self.dtype()));
}

} // namespace

Tensor empty_kernel(const std::vector<int64_t>& size, std::optional<DType> dtype,
                    std::optional<Device> device, bool pin_memory) {
    TP_CHECK(!pin_memory, "NPU tensors cannot be pinned host memory");
    numel_of(size);
    const int index = device ? index_of(*device) : current_device();
    return empty_on(size, dtype.value_or(DType::Float32), index);
}

Tensor empty_memory_format_kernel(const std::vector<int64_t>& size, std::optional<DType> dtype,
                                  std::optional<int64_t> layout, std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    (void)layout;
    (void)memory_format;
    return empty_kernel(size, dtype, device, pin_memory.value_or(false));
}

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    (void)non_blocking;
    if (self.numel() == 0) return self;
    const bool self_npu = self.device().is_privateuse1();
    const bool src_npu = src.device().is_privateuse1();
    TP_CHECK((self_npu || self.device().is_cpu()) && (src_npu || src.device().is_cpu()),
             "NPU copies run between NPU devices and the host, not ", src.device().toString(),
             " -> ", self.device().toString());
    if (self_npu && src_npu) {
        device_to_device(self, src);
    } else if (self_npu) {
        host_to_device(self, src);
    } else {
        device_to_host(self, src);
    }
    return self;
}

Tensor& fill_kernel(Tensor& self, const Scalar& value) {
    if (self.numel() == 0) return self;
    const int index = index_of(self.device());
    if (self.is_contiguous() && value.isIntegral(true) && value.to<int64_t>() == 0) {
        memset_on_device(self.data_ptr(), 0,
                         static_cast<size_t>(self.numel()) * elementSize(self.dtype()), index);
        return self;
    }
    Tensor host = host_tensor(self.sizes().vec(), self.dtype());
    {
        impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
        DispatchStub<Tensor&, Tensor&, const Scalar&>::call("fill_.Scalar", DispatchKey::CPU,
                                                            host, value);
    }
    host_to_device(self, host);
    return self;
}

Tensor& zero_kernel(Tensor& self) { return fill_kernel(self, Scalar(int64_t(0))); }

Tensor clone_kernel(const Tensor& self, std::optional<int64_t> memory_format) {
    (void)memory_format;
    Tensor out = empty_on(self.sizes().vec(), self.dtype(), index_of(self.device()));
    if (out.numel() > 0) strided_copy(out, self);
    return out;
}

Tensor contiguous_kernel(const Tensor& self, int64_t memory_format) {
    (void)memory_format;
    if (self.is_contiguous()) return self;
    return clone_kernel(self, std::nullopt);
}

Tensor empty_like_kernel(const Tensor& self, DType dtype, std::optional<Device> device) {
    const DType type = dtype == DType::Undefined ? self.dtype() : dtype;
    if (device && !device->is_privateuse1()) {
        return Tensor(self.sizes().vec(), type, *device);
    }
    const int index = device ? index_of(*device) : index_of(self.device());
    return empty_on(self.sizes().vec(), type, index);
}

Scalar item_kernel(const Tensor& self) {
    TP_CHECK(self.numel() == 1, "item() only supported for 1-element tensors");
    return to_host_dense(self).item();
}

namespace sig = kernel_signatures;

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, NPUCopyKernels) {
    m.impl("empty_like", static_cast<sig::empty_like*>(&empty_like_kernel));
    m.impl("item", static_cast<sig::item*>(&item_kernel));
    m.impl("_local_scalar_dense", static_cast<sig::_local_scalar_dense*>(&item_kernel));
    m.impl("empty", static_cast<sig::empty*>(&empty_kernel));
    m.impl("empty.memory_format",
           static_cast<sig::empty__memory_format*>(&empty_memory_format_kernel));
    m.impl("copy_", static_cast<sig::copy_*>(&copy_kernel));
    m.impl("fill_.Scalar", static_cast<sig::fill___Scalar*>(&fill_kernel));
    m.impl("zero_", static_cast<sig::zero_*>(&zero_kernel));
    m.impl("clone", static_cast<sig::clone*>(&clone_kernel));
    m.impl("contiguous", static_cast<sig::contiguous*>(&contiguous_kernel));
}

} // namespace npu
} // namespace tensorplay
