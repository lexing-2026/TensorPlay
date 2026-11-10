#include "AscendTensor.h"

#include <algorithm>

#include "Dispatcher.h"
#include "LocalDispatchKeySet.h"

namespace tensorplay {
namespace npu {

namespace {

struct Box {
    int64_t lo = 0;    // element offset of the box's first element from data_ptr()
    int64_t width = 1; // elements the box spans
};

// The smallest range of elements that holds every element of ``t``.  The
// offset of an element is the sum of one reach per dimension, so the range
// adds up the negative reaches below the first element and the positive ones
// above it; strides may be negative (a reversed view).
Box box_of(const Tensor& t) {
    Box box;
    int64_t hi = 0;
    const auto sizes = t.sizes().vec();
    const auto strides = t.strides();
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] <= 1) continue;
        const int64_t reach = (sizes[i] - 1) * strides[i];
        box.lo += std::min<int64_t>(0, reach);
        hi += std::max<int64_t>(0, reach);
    }
    box.width = hi - box.lo + 1;
    return box;
}

// A host tensor reading ``t``: the same sizes, strides and first element,
// over a host buffer holding the bytes of ``t``'s span.  Bytes between the
// elements (the gaps of a strided view) are carried along untouched.
Tensor mirror_on_host(const Tensor& t, int index) {
    const Box box = box_of(t);
    const size_t esz = elementSize(t.dtype());
    Tensor mirror(Storage(static_cast<size_t>(box.width) * esz), t.sizes().vec(), t.strides(),
                  t.dtype(), static_cast<size_t>(-box.lo));
    if (t.numel() > 0) {
        copy_to_host(static_cast<char*>(mirror.data_ptr()) + box.lo * static_cast<int64_t>(esz),
                     static_cast<const char*>(t.data_ptr()) +
                         box.lo * static_cast<int64_t>(esz),
                     static_cast<size_t>(box.width) * esz, index);
    }
    return mirror;
}

} // namespace

Tensor empty_on(const std::vector<int64_t>& sizes, DType dtype, int index) {
    int64_t numel = 1;
    for (int64_t d : sizes) numel *= d;
    const Device placed(DeviceType::PrivateUse1, index);
    Storage storage(static_cast<size_t>(numel) * elementSize(dtype), allocator(), placed);
    return Tensor(std::move(storage), sizes, dtype);
}

void host_copy_into(Tensor& dst, const Tensor& src) {
    impl::ExcludeDispatchKeyGuard below(DispatchKeySet::autograd_keys());
    DispatchStub<Tensor&, Tensor&, const Tensor&, bool>::call("copy_", DispatchKey::CPU, dst, src,
                                                              false);
}

void strided_copy(const Tensor& dst, const Tensor& src) {
    TP_CHECK(dst.dtype() == src.dtype(), "NPU strided_copy: dtype mismatch");
    TP_CHECK(dst.sizes().vec() == src.sizes().vec(), "NPU strided_copy: shape mismatch");
    if (dst.numel() == 0) return;
    const int index = index_of(dst.device());
    const size_t esz = elementSize(dst.dtype());
    if (dst.is_contiguous() && src.is_contiguous()) {
        copy_on_device(dst.data_ptr(), src.data_ptr(),
                       static_cast<size_t>(dst.numel()) * esz, index);
        return;
    }
    // The layout work happens on the host: both sides move as the byte box
    // their view can name, so a strided write keeps the bytes between the
    // elements it does not touch.
    const Box dst_box = box_of(dst);
    Tensor host_src = mirror_on_host(src, index);
    Tensor host_dst =
        dst.is_contiguous()
            ? Tensor(Storage(static_cast<size_t>(dst.numel()) * esz), dst.sizes().vec(),
                     dst.dtype())
            : mirror_on_host(dst, index);
    host_copy_into(host_dst, host_src);
    if (dst.is_contiguous()) {
        copy_to_device(dst.data_ptr(), host_dst.data_ptr(),
                       static_cast<size_t>(dst.numel()) * esz, index);
    } else {
        copy_to_device(static_cast<char*>(dst.data_ptr()) +
                           dst_box.lo * static_cast<int64_t>(esz),
                       static_cast<char*>(host_dst.data_ptr()) +
                           dst_box.lo * static_cast<int64_t>(esz),
                       static_cast<size_t>(dst_box.width) * esz, index);
    }
}

Tensor contiguous_on_device(const Tensor& t) {
    if (t.is_contiguous()) return t;
    Tensor out = empty_on(t.sizes().vec(), t.dtype(), index_of(t.device()));
    strided_copy(out, t);
    return out;
}

} // namespace npu
} // namespace tensorplay
