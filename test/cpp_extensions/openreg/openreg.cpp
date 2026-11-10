// openreg: a reference out-of-tree device backend.
//
// The smallest complete backend for the PrivateUse1 slot, written the way an
// extension for real hardware would be.  Its "devices" are host memory, so
// it runs anywhere, but nothing outside this file treats them as host
// memory: data crosses the boundary only through the allocator, the hooks
// and the backend's own copy kernel.
//
// What it registers:
//   * an allocator for DeviceType::PrivateUse1 (REGISTER_ALLOCATOR);
//   * device hooks: two devices, a per-thread current device, byte copies
//     (REGISTER_PRIVATEUSE1_HOOKS);
//   * native kernels for the operators every backend writes itself:
//     empty, empty.memory_format, copy_ and item;
//   * host fallbacks (CpuFallback.h) for the arithmetic a small training
//     step uses.  A real backend replaces these with device kernels one by
//     one.
//
// The device name ("openreg") is given from Python with
// tensorplay.utils.rename_privateuse1_backend.

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "Allocator.h"
#include "CpuFallback.h"
#include "Dispatcher.h"
#include "PrivateUse1Hooks.h"
#include "Tensor.h"
#include "tensorplay/ops/KernelSignatures.h"

namespace openreg {

using namespace tensorplay;

constexpr int64_t kDeviceCount = 2;

thread_local int64_t current_device_index = 0;

// Live allocations and the device each one belongs to.  A real backend asks
// its driver; the table lets copies check that pointers really are ours.
std::mutex allocations_lock;
std::unordered_map<void*, int64_t>& allocations() {
    static auto* table = new std::unordered_map<void*, int64_t>();
    return *table;
}

void release(void* data) {
    if (data == nullptr) return;
    {
        std::lock_guard<std::mutex> guard(allocations_lock);
        allocations().erase(data);
    }
    std::free(data);
}

class OpenRegAllocator final : public Allocator {
public:
    DataPtr allocate(size_t nbytes) const override {
        return allocate(nbytes, Device(DeviceType::PrivateUse1, current_device_index));
    }

    DataPtr allocate(size_t nbytes, const Device& device) const override {
        const int64_t index = device.index() < 0 ? current_device_index : device.index();
        TP_CHECK(index < kDeviceCount, "openreg has no device ", index);
        const Device placed(DeviceType::PrivateUse1, index);
        if (nbytes == 0) return DataPtr(nullptr, deleteNothing, placed);
        void* data = std::aligned_alloc(64, (nbytes + 63) / 64 * 64);
        TP_CHECK(data != nullptr, "openreg: out of memory allocating ", nbytes, " bytes");
        {
            std::lock_guard<std::mutex> guard(allocations_lock);
            allocations()[data] = index;
        }
        return DataPtr(data, release, placed);
    }
};

OpenRegAllocator allocator;
REGISTER_ALLOCATOR(DeviceType::PrivateUse1, &allocator);

struct OpenRegHooks final : PrivateUse1HooksInterface {
    int64_t device_count() const override { return kDeviceCount; }
    int64_t current_device() const override { return current_device_index; }
    void set_device(int64_t index) const override {
        TP_CHECK(index >= 0 && index < kDeviceCount, "openreg has no device ", index);
        current_device_index = index;
    }
    void copy_bytes(void* destination, const Device&, const void* source,
                    const Device&, size_t nbytes) const override {
        std::memcpy(destination, source, nbytes);
    }
    std::pair<size_t, size_t> memory_info(int64_t) const override {
        return {size_t(1) << 30, size_t(1) << 30};
    }
};

OpenRegHooks hooks;
REGISTER_PRIVATEUSE1_HOOKS(&hooks);

// ---------------------------------------------------------------------------
// Native kernels
// ---------------------------------------------------------------------------

int64_t device_index_of(std::optional<Device> device) {
    return device && device->index() >= 0 ? device->index() : current_device_index;
}

Tensor empty_kernel(const std::vector<int64_t>& size, std::optional<DType> dtype,
                    std::optional<Device> device, bool pin_memory) {
    TP_CHECK(!pin_memory, "openreg tensors cannot be pinned host memory");
    const DType type = dtype.value_or(DType::Float32);
    int64_t numel = 1;
    for (int64_t d : size) {
        TP_CHECK(d >= 0, "openreg: negative dimension ", d);
        numel *= d;
    }
    const Device placed(DeviceType::PrivateUse1, device_index_of(device));
    Storage storage(static_cast<size_t>(numel) * elementSize(type), &allocator, placed);
    return Tensor(std::move(storage), size, type);
}

Tensor empty_memory_format_kernel(const std::vector<int64_t>& size,
                                  std::optional<DType> dtype,
                                  std::optional<int64_t> layout,
                                  std::optional<Device> device,
                                  std::optional<bool> pin_memory,
                                  std::optional<int64_t> memory_format) {
    (void)layout;
    (void)memory_format;
    return empty_kernel(size, dtype, device, pin_memory.value_or(false));
}

// A host view of an openreg tensor's bytes.  Only this backend may form it:
// it knows its devices are addressable from the host.
Tensor host_alias(const Tensor& t) {
    const auto& impl = t.unsafeGetTensorImpl();
    const Storage& storage = impl->storage();
    Storage view(DataPtr(storage.data(), deleteNothing, Device(DeviceType::CPU)),
                 storage.nbytes());
    return Tensor(std::move(view), t.sizes().vec(), t.strides().vec(), t.dtype(),
                  impl->storage_offset());
}

Tensor& copy_kernel(Tensor& self, const Tensor& src, bool non_blocking) {
    (void)non_blocking;
    if (self.numel() == 0) return self;
    const bool self_ours = self.device().is_privateuse1();
    const bool src_ours = src.device().is_privateuse1();
    TP_CHECK((self_ours || self.device().is_cpu()) && (src_ours || src.device().is_cpu()),
             "openreg copies only between openreg devices and the host, not ",
             src.device().toString(), " -> ", self.device().toString());
    Tensor destination = self_ours ? host_alias(self) : self;
    Tensor source = src_ours ? host_alias(src) : src;
    // Both sides are host tensors now: the CPU copy handles strides,
    // broadcasting and dtype conversion.
    impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
    DispatchStub<Tensor&, Tensor&, const Tensor&, bool>::call(
        "copy_", DispatchKey::CPU, destination, source, false);
    return self;
}

Scalar item_kernel(const Tensor& self) {
    return host_alias(self).item();
}

} // namespace openreg

namespace tensorplay {

using namespace openreg;
namespace sig = kernel_signatures;

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, OpenRegNative) {
    m.impl("empty", static_cast<sig::empty*>(&empty_kernel));
    m.impl("empty.memory_format",
           static_cast<sig::empty__memory_format*>(&empty_memory_format_kernel));
    m.impl("copy_", static_cast<sig::copy_*>(&copy_kernel));
    m.impl("item", static_cast<sig::item*>(&item_kernel));
}

#define OPENREG_HOST_FALLBACK(name, symbol) \
    m.impl(name, &cpu_fallback::kernel<name, sig::symbol>)

TENSORPLAY_LIBRARY_IMPL(PrivateUse1, OpenRegHostFallbacks) {
    OPENREG_HOST_FALLBACK("fill_.Scalar", fill___Scalar);
    OPENREG_HOST_FALLBACK("zero_", zero_);
    OPENREG_HOST_FALLBACK("add.Tensor", add__Tensor);
    OPENREG_HOST_FALLBACK("add_.Tensor", add___Tensor);
    OPENREG_HOST_FALLBACK("sub.Tensor", sub__Tensor);
    OPENREG_HOST_FALLBACK("mul.Tensor", mul__Tensor);
    OPENREG_HOST_FALLBACK("mul.Scalar", mul__Scalar);
    OPENREG_HOST_FALLBACK("div.Tensor", div__Tensor);
    OPENREG_HOST_FALLBACK("neg", neg);
    OPENREG_HOST_FALLBACK("mm", mm);
    OPENREG_HOST_FALLBACK("addmm", addmm);
    OPENREG_HOST_FALLBACK("relu", relu);
    OPENREG_HOST_FALLBACK("threshold_backward", threshold_backward);
    OPENREG_HOST_FALLBACK("sum", sum);
    OPENREG_HOST_FALLBACK("sum.dim_IntList", sum__dim_IntList);
    OPENREG_HOST_FALLBACK("mean", mean);
    OPENREG_HOST_FALLBACK("clone", clone);
}

#undef OPENREG_HOST_FALLBACK

} // namespace tensorplay
