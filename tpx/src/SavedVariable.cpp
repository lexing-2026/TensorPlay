#include "SavedVariable.h"

#include "Exception.h"

namespace tensorplay {
namespace tpx {

void SavedVariable::save(const Tensor& tensor) {
    if (!tensor.defined()) {
        data_ = Tensor();
        saved_version_ = 0;
        return;
    }
    data_ = tensor;
    saved_version_ = tensor.unsafeGetTensorImpl()->version();
}

Tensor SavedVariable::unpack() const {
    if (!data_.defined()) return Tensor();
    uint32_t current = data_.unsafeGetTensorImpl()->version();
    if (current != saved_version_) {
        TP_THROW(RuntimeError,
                 "one of the variables needed for gradient computation has "
                 "been modified by an inplace operation: [saved version: " +
                     std::to_string(saved_version_) +
                     "; current version: " + std::to_string(current) + "]");
    }
    return data_;
}

} // namespace tpx
} // namespace tensorplay
