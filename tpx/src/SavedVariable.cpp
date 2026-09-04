#include "SavedVariable.h"

#include "Exception.h"

#include <utility>
#include <vector>

namespace tensorplay {
namespace tpx {

namespace {
thread_local std::vector<std::shared_ptr<SavedVariableHooks>> g_hooks;
}

std::shared_ptr<SavedVariableHooks> current_saved_variable_hooks() {
    if (g_hooks.empty()) return nullptr;
    return g_hooks.back();
}

void push_saved_variable_hooks(std::shared_ptr<SavedVariableHooks> hooks) {
    g_hooks.push_back(std::move(hooks));
}

void pop_saved_variable_hooks() {
    if (g_hooks.empty()) {
        TP_THROW(RuntimeError, "saved variable hook stack is empty");
    }
    g_hooks.pop_back();
}

void SavedVariable::save(const Tensor& tensor) {
    if (!tensor.defined()) {
        data_ = Tensor();
        packed_.reset();
        hooks_.reset();
        saved_version_ = 0;
        return;
    }
    hooks_ = current_saved_variable_hooks();
    data_ = tensor;
    saved_version_ = tensor.unsafeGetTensorImpl()->version();
    if (hooks_) {
        packed_ = hooks_->pack(tensor);
        data_ = Tensor();
    } else {
        packed_.reset();
    }
}

Tensor SavedVariable::unpack() const {
    if (hooks_) {
        return hooks_->unpack(packed_);
    }
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
