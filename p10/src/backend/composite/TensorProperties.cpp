// Composite kernels: is_same_size / get_device.

#include "Tensor.h"
#include "Dispatcher.h"

#include <cstdint>

namespace tensorplay {
namespace composite {

bool is_same_size_native(const Tensor& self, const Tensor& other) {
    return self.shape() == other.shape();
}

int64_t get_device_native(const Tensor& self) {
    return self.device().index();
}

// bool counts as signed (its values are 0/1), unsigned integer widths are not.
bool is_signed_native(const Tensor& self) {
    switch (self.dtype()) {
        case DType::UInt8:
        case DType::UInt16:
        case DType::UInt32:
        case DType::UInt64:
            return false;
        default:
            return true;
    }
}

TENSORPLAY_LIBRARY_IMPL(Composite, TensorPropertiesComposite) {
    m.impl("is_same_size", is_same_size_native);
    m.impl("get_device", get_device_native);
    m.impl("is_signed", is_signed_native);
}

} // namespace composite
} // namespace tensorplay
